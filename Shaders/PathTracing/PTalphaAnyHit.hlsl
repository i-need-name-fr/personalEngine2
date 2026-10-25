#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RaytracingAccelerationStructure TLAS;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(3 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(4 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(5 , 0)]] ConstantBuffer<SunSource> sunBuffer;


[[vk::binding(6 , 0)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(7 , 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(8 , 0)]] RWStructuredBuffer<InstanceData> instanceLayout;
[[vk::binding(9 , 0)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(10 , 0)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(11 , 0)]] RWStructuredBuffer<LUT> LUT_Buffer;

[[vk::binding(12 , 0)]] RWStructuredBuffer<ReservoirDI> ReSTIRDIBuffer;
[[vk::binding(13 , 0)]] RWStructuredBuffer<ReservoirDI> prevReSTIRDIBuffer;
[[vk::binding(14 , 0)]] RWStructuredBuffer<ReservoirGI> ReSTIRGIBuffer;
[[vk::binding(15 , 0)]] RWStructuredBuffer<ReservoirGI> prevReSTIRGIBuffer;

[[vk::binding(16 , 0)]] TextureCube skybox;
[[vk::binding(17 , 0)]] SamplerState samp;
[[vk::binding(18 , 0)]] Texture2D transmissionLUT;
[[vk::binding(19 , 0)]] Texture1D multiScatter;

[[vk::image_format("rgba16f")]]
[[vk::binding(20 , 0)]] RWTexture2D<float4> diffuseImage;

[[vk::image_format("rgba16")]]
[[vk::binding(21 , 0)]] RWTexture2D<float4> specularImage;

[[vk::image_format("rgba16")]]
[[vk::binding(22 , 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("rgba16")]]
[[vk::binding(23 , 0)]] RWTexture2D<float4> indirectImage;

[[vk::image_format("rgba16")]]
[[vk::binding(24 , 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(25 , 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("r32f")]]
[[vk::binding(26 , 0)]] RWTexture2D<float> roughnessImage;

[[vk::image_format("r32f")]]
[[vk::binding(27 , 0)]] RWTexture2D<float> metallicImage;

[[vk::image_format("r32f")]]
[[vk::binding(28 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("rg32f")]]
[[vk::binding(29 , 0)]] RWTexture2D<float2> shadowImage;

[[vk::image_format("rg32f")]]
[[vk::binding(30 , 0)]] RWTexture2D<float2> prevShadowImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(31 , 0)]] RWTexture2D<float4> viewImage;

[[vk::image_format("rg32f")]]
[[vk::binding(32 , 0)]] RWTexture2D<float2> motionVector;

[[vk::image_format("r32f")]]
[[vk::binding(33 , 0)]] RWTexture2D<float> pixelCheck;


// set 1 : the textures of the TextureManager2 ( the alpha of the base colour map )
[[vk::binding(0 , 1)]] StructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState typicalSampler;
[[vk::binding(2 , 1)]] Texture2D allTextures[];

#define ALPHA_TEST_SAMPLER typicalSampler
#include "../../include/ShaderStructures/AlphaTest.h"

struct [raypayload] RayPayLoad_primary_
{
    uint seed : write(caller , closesthit , miss) : read(caller , closesthit , miss);
};

// the any hit of a hit group of the geometry : the pixels of a leaf under the cutoff of the alpha test do not exist for the ray ( the hit is ignored ) , everything else is
// accepted as it is ( the instances that are not tested are opaque for the ray , this shader does not run for them , except the bodies of the lamps )
[shader("anyhit")]
void main(inout RayPayLoad_primary_ payload , in BuiltInTriangleIntersectionAttributes attr){
    if(!passesAlphaTest(InstanceID() , PrimitiveIndex() , attr.barycentrics)){
        IgnoreHit();
    }
}
