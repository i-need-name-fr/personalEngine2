#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RaytracingAccelerationStructure TLAS;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(3 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(4 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(5 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(6 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(7 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;

[[vk::binding(9 , 0)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(10 , 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(11 , 0)]] RWStructuredBuffer<InstanceData> instanceLayout;
[[vk::binding(12 , 0)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(14 , 0)]] RWStructuredBuffer<LUT> LUT_Buffer;

[[vk::binding(15, 0)]] TextureCube skybox;
[[vk::binding(16, 0)]] SamplerState samp;
[[vk::binding(17 , 0)]] Texture2D transmissionLUT;
[[vk::binding(18, 0)]] Texture1D multiScatter;

[[vk::binding(0 , 1)]] StructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState textureSampler;
[[vk::binding(2 , 1)]] Texture2D allTextures[];

[[vk::binding(0 , 2)]] RWStructuredBuffer<RC::ProbeInfo> RCProbeBuffer;
[[vk::binding(1 , 2)]] RWStructuredBuffer<RC::CascadeInfo> RCCascadesBuffer;
[[vk::binding(2 , 2)]] ConstantBuffer<RC::GeneralProbeInfo> RCProbeInfo;
[[vk::image_format("rgba16f")]]
[[vk::binding(3 , 2)]] RWTexture2D<float4> RCImages[];

[[vk::push_constant]] struct {
    uint id;
} push;

void recordData(const float2 buildUV , const float4 color , const uint2 probeIndex , const uint casID){
    // idea of this function is to record 
    // so we have to find the cascade we are going to be using
    const RC::CascadeInfo cas_ = RCCascadesBuffer[casID];
    // since now we have the cas , we can find out in which localTexel of 
    // probe ( as an page ) we are going to be writing in

    // will be changed
    if(any(buildUV >= 1.0f) || any(buildUV <= 0.0f)) return;
    // now we have to multiply the UV with build size of current cascade
    const uint2 buildPixel = uint2(buildUV * float2(cas_.BUILD_SIZE , cas_.BUILD_SIZE));
    // now we need to locate our probe in RCImage
    const uint probePerRow = cas_.PROBE_PER_ROW;
    const uint2 pixel = (probeIndex * cas_.BUILD_SIZE) + buildPixel;
    // the cascade 0 is written to the image with the parity of the frame ( ping pong ) , the others to their own images ( RC::imageIndex )
    RCImages[RC::imageIndex(casID , generalBuffer.frameIndex)][pixel] = color;
}

struct [raypayload] RayPayLoad{
    float3 finalColor : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint seed : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint cascadeID : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float alpha : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float hitDistance : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint mode : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

[shader("raygeneration")]
void main(){
    uint2 id = DispatchRaysIndex().xy;
    // idea is like this
    // id is a texel coord of current RC image
    const RC::CascadeInfo cas_ = RCCascadesBuffer[push.id];
    const uint2 cas_R = uint2(cas_.CASCADE_X , cas_.CASCADE_Y);
    if(any(id.xy >= cas_R)) return;
    // probe coord is pretty simple i think 
    const uint2 probeCoord = id.xy / uint(cas_.BUILD_SIZE);
    const int2 buildTexel = id.xy - (probeCoord * uint(cas_.BUILD_SIZE));
    // since if we overwrite the board  , it will cause artefacts
    float2 buildUV = float2(buildTexel+ 0.5f) / float(cas_.BUILD_SIZE);
    const float3 dir = octahedralDecode(buildUV);

    const uint probeID = probeCoord.y * cas_.PROBE_PER_ROW + probeCoord.x + cas_.PROBE_OFFSET; 
    if(probeID >= RCProbeInfo.TOTAL_PROBES) return;
    const RC::ProbeInfo probe = RCProbeBuffer[probeID];

    // the rays of a texel : 1 << cascade , every one goes to a random point of the CAP of the texel ( the solid angle of a texel of the map is 4 pi / BUILD_SIZE^2 , the cap
    // 2 pi ( 1 - cos a ) has the same area : cos a = 1 - 2 / BUILD_SIZE^2 ) . coneSamplingVector takes the cap as sin a = r / max( d , r ) , so d = 1 and r = sin a
    const uint raysCount = 1u << push.id;
    const float cosA = 1.0f - 2.0f / (float(cas_.BUILD_SIZE) * float(cas_.BUILD_SIZE));
    const float sinA = sqrt(saturate(1.0f - cosA * cosA));

    // the HITS and the MISSES are averaged apart : a ray that left the interval has the colour of the sky as a placeholder ( RCmiss ) , it must not get into the colour of the hits
    float3 hitColor = 0.0f;
    float hitWeight = 0.0f;
    float hitDistanceSum = 0.0f;
    float3 missColor = 0.0f;
    float missWeight = 0.0f;
    float missDistanceSum = 0.0f;
    uint missCount = 0;

    RayPayLoad load = (RayPayLoad)0;
    load.seed = id.x * 3942u + id.y * 3413u + generalBuffer.frameIndex * 9283u;
    load.cascadeID = push.id;

    [loop]
    for(uint i = 0 ; i < raysCount ; i++)
    {
        load.seed += 2953u * i;
        const float3 nextDir = coneSamplingVector(load.seed , dir , 1.0f , sinA);
        const float dirWeight = pow(saturate(dot(nextDir , dir)) , 5.0f);

        RayDesc ray;
        ray.Direction = nextDir;
        ray.Origin = probe.position;
        ray.TMax = cas_.INTERVAL_END;
        ray.TMin = max(cas_.INTERVAL_START , 0.0001f);

        TraceRay(TLAS , RAY_FLAG_NONE , 0xFF , 0 , 0 , 0 , ray, load);

        // NaN and Inf of a ray are not stored , and one very bright ray is limited ( a firefly runs through the cache otherwise )
        const float3 rayColor = sanitizeRadiance(load.finalColor , RC::CACHE_MAX_RADIANCE);

        if(load.alpha == 0.0f)
        {
            missCount++;
            missColor += rayColor * dirWeight;
            missWeight += dirWeight;
            missDistanceSum += load.hitDistance * dirWeight;
        }
        else
        {
            hitColor += rayColor * dirWeight;
            hitWeight += dirWeight;
            hitDistanceSum += load.hitDistance * dirWeight;
        }
    }

    // the texel is a hit if the rays that missed are not the majority ( then the colour is the one of the hits ) , otherwise it is a miss ( the colour of the sky , the distance is
    // the end of the interval ) . the alpha is packed ( RC::packHit ) : the hit flag and the distance
    const bool texelHit = (float(missCount) <= float(raysCount) * 0.5f) && (hitWeight > 0.0f);
    float3 finalColor;
    float finalDistance;
    if(texelHit)
    {
        finalColor = hitColor / max(hitWeight , 1e-20f);
        finalDistance = hitDistanceSum / max(hitWeight , 1e-20f);
    }
    else
    {
        finalColor = missColor / max(missWeight , 1e-20f);
        finalDistance = missDistanceSum / max(missWeight , 1e-20f);
    }

    recordData(buildUV , float4(finalColor , RC::packHit(texelHit , finalDistance)) , probeCoord , push.id);
}