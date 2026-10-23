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
    RCImages[casID][pixel] = color;
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

    RayDesc ray;
    ray.Origin = probe.position;
    ray.Direction = dir;
    ray.TMax = 1000.0f;
    ray.TMin = 0.0001f;

    RayPayLoad load = (RayPayLoad)0;
    load.seed = id.x * 3942u + id.y * 3413u + generalBuffer.frameIndex * 9283u;
    load.cascadeID = push.id;
    TraceRay(TLAS , RAY_FLAG_NONE , 0xFF , 0 , 0 , 0 , ray, load);
    const float4 result = float4(load.finalColor , RC::packHit(load.alpha >= 0.5f , load.hitDistance));

    recordData(buildUV , result , probeCoord , push.id);
}