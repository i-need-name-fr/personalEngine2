// the descriptor sets ( the same in task , mesh and fragment )
#include "meshPassBindings.hlsli"

struct VsOutput{
    float4 clipPos  : SV_Position;
    [[vk::location(0)]] float3 fragPos                    : TEXCOORD0;
    [[vk::location(1)]] float3 normal                     : TEXCOORD1;
    [[vk::location(2)]] float3x3 TBN                      : TEXCOORD2;
    [[vk::location(5)]] float2 uv                         : TEXCOORD5;
    [[vk::location(6)]] nointerpolation uint LUTindex     : TEXCOORD6;
};

struct VsInput{
    // general vertex
    float3 position  ;
    float3 normal    ;
    float3 tangent   ;
    float3 bitangent ;
    float2 uv        ;

};

VsOutput processVertex(uint instanceID , const VsInput input)
{
    VsOutput output;

    InstanceData currentInstance = instanceLayoutBuffer[instanceID];
    uint meshID = currentInstance.meshDataIndex;
    uint instanceBufferIndex = meshBuffer[meshID].instanceIndex;
    float4x4 model = instanceBuffer[instanceBufferIndex];
    uint LUT_index = currentInstance.LUT_index;

    output.fragPos = mul(model , float4(input.position , 1.0f)).xyz;
    output.normal = normalize(mul(normalMatrix3x3((float3x3)model) , input.normal));
    output.LUTindex = LUT_index;
    output.uv = input.uv;
    const float3 T = normalize(mul(model , float4(input.tangent , 0.0f)).xyz);
    const float3 B = normalize(mul(model , float4(input.bitangent , 0.0f)).xyz);
    output.TBN = float3x3(T , B , output.normal);
    output.clipPos = mul(generalBuffer.viewProj , float4(output.fragPos , 1.0f));
    return output;
}

static const uint TASK_GROUP_SIZE = 32;

struct Payload
{
    uint layoutIndex;
    uint meshletID[TASK_GROUP_SIZE];
};

struct PrimOutput
{
    bool cull : SV_CullPrimitive;
};

groupshared float4 sClip[ms::MAX_BOUND_VERTEX_COUNT];

bool tooSmall(const float4 c0 ,  const float4 c1 , const float4 c2)
{
    if(c0.w < 0.0f) return false;
    if(c1.w < 0.0f) return false;
    if(c2.w < 0.0f) return false;

    const float2 p0 = c0.xy / c0.w;
    const float2 p1 = c1.xy / c1.w;
    const float2 p2 = c2.xy / c2.w;

    const float2 texelSize = float2(float(generalBuffer.screenWidth) , float(generalBuffer.screenHeight));

    const float2 uv1 = (p0 * 0.5f + 0.5f) * texelSize;
    const float2 uv2 = (p1 * 0.5f + 0.5f) * texelSize;
    const float2 uv3 = (p2 * 0.5f + 0.5f) * texelSize;

    float2 boxMax = -1e9f;
    float2 boxMin =  1e9f;
    
    float2 uvs[3] = {uv1 , uv2 , uv3};

    [unroll]
    for(uint i = 0 ; i < 3 ; i++)
    {
        boxMax.x = max(boxMax.x , uvs[i].x);
        boxMax.y = max(boxMax.y , uvs[i].y);

        boxMin.x = min(boxMin.x , uvs[i].x);
        boxMin.y = min(boxMin.y , uvs[i].y);
    }

    return any(round(boxMin) == round(boxMax));
}

bool cullTriangle(const float4 c0 , const float4 c1 , const float4 c2)
{   
    if(c0.w < 0.0f) return false;
    if(c1.w < 0.0f) return false;
    if(c2.w < 0.0f) return false;

    if(tooSmall(c0 , c1 , c2)) return true;

    const float det =
        c0.x * (c1.y * c2.w - c2.y * c1.w)
        - c0.y * (c1.x * c2.w - c2.x * c1.w)
        + c0.w * (c1.x * c2.y - c2.x * c1.y);
    return det < 0.0f;

}

[shader("mesh")]
[numthreads(128 , 1 , 1)]
[outputtopology("triangle")]
void main(uint3 id : SV_DispatchThreadID , uint gtid : SV_GroupThreadID, uint gid : SV_GroupID , in payload Payload load, out vertices VsOutput output[ms::MAX_BOUND_VERTEX_COUNT] , out indices uint3 ind[ms::MAX_BOUND_TRIANGLE_COUNT] , out primitives PrimOutput prims[ms::MAX_BOUND_TRIANGLE_COUNT])
{
    if(gid < TASK_GROUP_SIZE) 
    {   

        const ms::Meshlet m = meshletBuffer[load.meshletID[gid]];
        SetMeshOutputCounts(m.vertexCount, m.indexCount);
        const MeshData mesh = meshBuffer[instanceLayoutBuffer[load.layoutIndex].meshDataIndex];
        if(gtid < m.vertexCount)
        {
            // now we process each triangle and also start setting indices
            VsInput input = (VsInput)0;
            const Vertex v = vertices[meshletVertices[m.firstVertex + gtid] + mesh.firstVertex];
            input.position = v.position;
            input.normal = v.normal;
            input.bitangent = v.bitangent;
            input.tangent = v.tangent;
            input.uv = v.UV;
            const VsOutput processed = processVertex(load.layoutIndex , input);
            output[gtid] = processed;
            sClip[gtid] = processed.clipPos;
        }
        GroupMemoryBarrierWithGroupSync();
        if(gtid < (m.indexCount))
        {
            const uint i0 = meshletIndices[m.firstIndex + gtid * 3 ];
            const uint i1 = meshletIndices[m.firstIndex + gtid * 3 +1];
            const uint i2 = meshletIndices[m.firstIndex + gtid * 3 +2];
            ind[gtid] = uint3(i0 , i1 , i2);
            prims[gtid].cull = cullTriangle(sClip[i0] , sClip[i1] , sClip[i2]);
        }

    }
}