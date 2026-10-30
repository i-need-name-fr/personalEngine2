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

[shader("mesh")]
[numthreads(128 , 1 , 1)]
[outputtopology("triangle")]
void main(uint3 id : SV_DispatchThreadID , uint gtid : SV_GroupThreadID, uint gid : SV_GroupID , in payload Payload load, out vertices VsOutput output[ms::MAX_BOUND_VERTEX_COUNT] , out indices uint3 ind[ms::MAX_BOUND_TRIANGLE_COUNT] )
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
            output[gtid] = processVertex(load.layoutIndex , input);
        }
        if(gtid < (m.indexCount))
        {
            const uint i0 = meshletIndices[m.firstIndex + gtid * 3 ];
            const uint i1 = meshletIndices[m.firstIndex + gtid * 3 +1];
            const uint i2 = meshletIndices[m.firstIndex + gtid * 3 +2];
            ind[gtid] = uint3(i0 , i1 , i2);
        }

    }
}