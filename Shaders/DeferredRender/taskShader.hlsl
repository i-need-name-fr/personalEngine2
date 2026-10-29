#include "../../include/ShaderStructures/ShaderStructure.h"

// ONE descriptor set for the whole mesh pass ( task + mesh + fragment ) : 0 .. 4 are the same as in the old deferred render ( the fragment shader reads 0 and 4 ) , 5 .. 10 are the meshlets
[[vk::binding(0 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(1 , 0)]] RWStructuredBuffer<InstanceData> instanceLayoutBuffer;
[[vk::binding(2 , 0)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(3 , 0)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(5 , 0)]] RWStructuredBuffer<ms::GeometryData> geometryBuffer;
[[vk::binding(6 , 0)]] RWStructuredBuffer<ms::Meshlet> meshletBuffer;
[[vk::binding(7 , 0)]] RWStructuredBuffer<ms::MeshletBound> boundBuffer;
[[vk::binding(8 , 0)]] RWStructuredBuffer<uint> meshletVertices;
[[vk::binding(9 , 0)]] RWStructuredBuffer<uint> meshletIndices;

static const uint TASK_GROUP_SIZE = 32;

struct Payload
{
    uint layoutIndex;
    uint meshletID[TASK_GROUP_SIZE];
};

groupshared Payload sPayload;

[shader("amplification")]
[numthreads(TASK_GROUP_SIZE , 1 , 1)]
void main(uint3 groupID : SV_GroupID , uint threadIndex : SV_GroupIndex)
{
    const uint layoutIndex = groupID.y;
    const InstanceData layout = instanceLayoutBuffer[layoutIndex];

    bool visible = false;
    uint meshletID = 0;

    if(layout.needsToBeDraw)
    {
        const MeshData mesh = meshBuffer[layout.meshDataIndex];
        const ms::GeometryData geom = geometryBuffer[mesh.geometryIndex];
        const uint localIndex = groupID.x * TASK_GROUP_SIZE + threadIndex;

        if(localIndex < geom.meshletCount)
        {
            meshletID = geom.meshletOffset + localIndex;

            const float4x4 model = instanceBuffer[mesh.instanceIndex];
            const ms::MeshletBound bound = boundBuffer[meshletID];

            const float3 center = mul(model , float4(bound.sphere.xyz , 1.0f)).xyz;
            const float3 axis = normalize(mul(model , float4(bound.cone.xyz , 0.0f)).xyz);
            const float scaleX = length(float3(model[0][0] , model[1][0] , model[2][0]));
            const float scaleY = length(float3(model[0][1] , model[1][1] , model[2][1]));
            const float scaleZ = length(float3(model[0][2] , model[1][2] , model[2][2]));
            const float radius = bound.sphere.w * max(scaleX , max(scaleY , scaleZ));
            const float cutoff = bound.cone.w;

            const float3 toMeshlet = center - generalBuffer.playerPosition;
            const bool backFacing = dot(toMeshlet , axis) >= cutoff * length(toMeshlet) + radius;

            visible = !backFacing;
        }
    }

    if(visible)
    {
        sPayload.meshletID[WavePrefixCountBits(visible)] = meshletID;
    }
    if(threadIndex == 0)
    {
        sPayload.layoutIndex = layoutIndex;
    }

    DispatchMesh(WaveActiveCountBits(visible) , 1 , 1 , sPayload);
}
