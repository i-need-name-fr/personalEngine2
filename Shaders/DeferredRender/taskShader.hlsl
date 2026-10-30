// the descriptor sets ( the same in task , mesh and fragment )
#include "meshPassBindings.hlsli"

static const uint TASK_GROUP_SIZE = 32;

struct Payload
{
    uint layoutIndex;
    uint meshletID[TASK_GROUP_SIZE];
};

void GetAABBCorners(float3 boxMin, float3 boxMax , inout float3 corners[8]) {

    corners[0] = float3(boxMin.x, boxMin.y, boxMin.z);
    corners[1] = float3(boxMax.x, boxMin.y, boxMin.z);
    corners[2] = float3(boxMin.x, boxMax.y, boxMin.z);
    corners[3] = float3(boxMax.x, boxMax.y, boxMin.z);
    corners[4] = float3(boxMin.x, boxMin.y, boxMax.z);
    corners[5] = float3(boxMax.x, boxMin.y, boxMax.z);
    corners[6] = float3(boxMin.x, boxMax.y, boxMax.z);
    corners[7] = float3(boxMax.x, boxMax.y, boxMax.z);

}

uint computeMipLevel(float2 screenMin , float2 screenMax , uint2 baseSize , uint maxMip ){
    float2 sizeInPixels = (screenMax - screenMin) * float2(baseSize);
    float maxDimensions = max(sizeInPixels.x ,sizeInPixels.y );

    if(maxDimensions < 1.0f) return 0;

    const uint mip = uint(ceil(log2(maxDimensions)));
    return min(mip , maxMip);
}

uint maxMips(uint width , uint height){
    return (uint)(floor(log2(max(width , height))));
}

bool ProjectAABBToScreen(float3 boxMax , float3 boxMin , float4x4 vp , out float2 screenMin , out float2 screenMax , out float objectClosestDepth){
    objectClosestDepth = 1e9;
    screenMin = float2(1e9 , 1e9);
    screenMax = float2(-1e9 , -1e9);

    float3 corners[8];
    GetAABBCorners(boxMin  ,boxMax , corners);

    for(uint i = 0 ; i < 8 ; i++){

        const float4 clipPos = mul(vp , float4(corners[i] , 1.0f));

        if(clipPos.w <= 0.0001f) return false;

        float3 ndc = clipPos.xyz / clipPos.w;

        ndc.y = -ndc.y;

        float2 screenPos = ndc.xy * 0.5f + 0.5f;

        screenMin = min(screenPos , screenMin);
        screenMax = max(screenPos , screenMax);
        objectClosestDepth = min(objectClosestDepth , ndc.z);
    }

    screenMax = saturate(screenMax);
    screenMin = saturate(screenMin);

    return true;
}


static const int MAX_OCCLUSION_TEXELS = 4;
static const float OCCLUSION_BIAS = 1e-6f;

bool needsToBeOccluded(MeshData object){

    float2 screenMax , screenMin;
    float objectClosestDepth;

    if(!ProjectAABBToScreen(object.boxMax , object.boxMin , generalBuffer.prevViewProj , screenMin , screenMax , objectClosestDepth)) return false;

    uint width , height , levels;
    prevDepth.GetDimensions(0 , width , height , levels);

    const uint mip = computeMipLevel(screenMin , screenMax , uint2(width , height) , levels - 1);

    const int2 mipSize = int2(max(1u , width >> mip) , max(1u , height >> mip));
    const int2 lo = clamp(int2(floor(screenMin * float2(mipSize))) , int2(0 , 0) , mipSize - 1);
    const int2 hi = clamp(int2(floor(screenMax * float2(mipSize))) , int2(0 , 0) , mipSize - 1);

    if(any((hi - lo) >= MAX_OCCLUSION_TEXELS)) return false;

    float occDepth = 0.0f;
    for(int y = lo.y ; y <= hi.y ; y++)
    {
        for(int x = lo.x ; x <= hi.x ; x++)
        {
            occDepth = max(occDepth , prevDepth.Load(int3(x , y , mip)).x);
        }
    }

    return objectClosestDepth > occDepth + OCCLUSION_BIAS;
}

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

        if(localIndex < geom.meshletCount && !needsToBeOccluded(mesh))
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
