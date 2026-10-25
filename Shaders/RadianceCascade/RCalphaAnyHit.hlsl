#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RaytracingAccelerationStructure TLAS;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(3 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(4 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(5 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(6 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(7 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(8 , 0)]] RWStructuredBuffer<lg::LightSource> lightBuffer;

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

#define ALPHA_TEST_SAMPLER textureSampler
#include "../../include/ShaderStructures/AlphaTest.h"

struct [raypayload] RayPayLoad{
    float3 finalColor : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint seed : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint cascadeID : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float alpha : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float hitDistance : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint mode : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

// the any hit of a hit group of the geometry : the pixels of a leaf under the cutoff of the alpha test do not exist for the ray ( the hit is ignored ) , everything else is
// accepted as it is ( the instances that are not tested are opaque for the ray , this shader does not run for them , except the bodies of the lamps )
[shader("anyhit")]
void main(inout RayPayLoad payload , in BuiltInTriangleIntersectionAttributes attr){
    if(!passesAlphaTest(InstanceID() , PrimitiveIndex() , attr.barycentrics)){
        IgnoreHit();
    }
}
