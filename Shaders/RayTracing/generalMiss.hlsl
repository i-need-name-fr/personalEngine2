#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RaytracingAccelerationStructure TLAS;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(3 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(4 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(5 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(6 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(7 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(8 , 0)]] RWStructuredBuffer<LightSource> lightBuffer;

[[vk::binding(9 , 0)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(10 , 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(11 , 0)]] RWStructuredBuffer<InstanceData> instanceLayout;
[[vk::binding(12 , 0)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(14 , 0)]] RWStructuredBuffer<LUT> LUT_Buffer;

[[vk::binding(15 , 0)]] RWStructuredBuffer<ReservoirDI> ReSTIRDIBuffer;
[[vk::binding(16 , 0)]] RWStructuredBuffer<ReservoirDI> prevReSTIRDIBuffer;
[[vk::binding(17 , 0)]] RWStructuredBuffer<ReservoirGI> ReSTIRGIBuffer;
[[vk::binding(18, 0)]] RWStructuredBuffer<ReservoirGI> prevReSTIRGIBuffer;

[[vk::binding(19, 0)]] TextureCube skybox;
[[vk::binding(20, 0)]] SamplerState samp;
[[vk::binding(21 , 0)]] Texture2D transmissionLUT;
[[vk::binding(22, 0)]] Texture1D multiScatter;

[[vk::image_format("rgba16f")]]
[[vk::binding(23, 0)]] RWTexture2D<float4> diffuseImage;

[[vk::image_format("rgba16")]]
[[vk::binding(24, 0)]] RWTexture2D<float4> specularImage;

[[vk::image_format("rgba16")]]
[[vk::binding(25, 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("rgba16")]]
[[vk::binding(26, 0)]] RWTexture2D<float4> indirectImage;

[[vk::image_format("rgba16")]]
[[vk::binding(27, 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(28, 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("r32f")]]
[[vk::binding(29, 0)]] RWTexture2D<float> roughnessImage;

[[vk::image_format("r32f")]]
[[vk::binding(30, 0)]] RWTexture2D<float> metallicImage;

[[vk::image_format("r32f")]]
[[vk::binding(31, 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("r32f")]]
[[vk::binding(32, 0)]] RWTexture2D<float> shadowImage;

[[vk::image_format("r32f")]]
[[vk::binding(33, 0)]] RWTexture2D<float> prevShadowImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(34, 0)]] RWTexture2D<float4> viewImage;

[[vk::image_format("rg32f")]]
[[vk::binding(35, 0)]] RWTexture2D<float2> motionVector;

[[vk::image_format("r32f")]]
[[vk::binding(36, 0)]] RWTexture2D<float> pixelCheck;


struct LightData{
    float3 diffuse;
    float3 specular;
    float3 indirect;
    float shadow;
    // the REAL distances ( not weights ) : the denoisers take the radius of the filter from them . diffuse and indirect share the first one
    float diffuseHitDistance;
    float specularHitDistance;
};

struct PixelData{
    float3 position;
    float3 normal;
    float3 albedo;
    float3 viewDir;
    float2 motionVector;
    float depth;
    float metallic;
    float roughness;
    float pixelType;
};

struct [raypayload] RayPayLoad{
    uint depth : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    uint seed : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    PixelData pixelInfo : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    LightData lightInfo : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    bool fromReflection : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
};

[shader("miss")]
void main(inout RayPayLoad load){
    const float3 dir = WorldRayDirection();
    const uint2 id = DispatchRaysIndex().xy;
    const float3 color = skybox.SampleLevel(samp , dir , 0).xyz;
    load.lightInfo.diffuse = color;
    load.lightInfo.specular = float3(0.0f ,0.0f , 0.0f);
    load.lightInfo.indirect = float3(0.0f ,0.0f , 0.0f);
    load.pixelInfo.pixelType = SKYBOX_PIXEL;
    load.lightInfo.shadow = 1.0f;
    // the sky is far away : the radius of the filter of a far hit is the biggest one
    load.lightInfo.diffuseHitDistance = 10000.0f;
    load.lightInfo.specularHitDistance = 10000.0f;

    load.pixelInfo.normal = -dir;
    load.pixelInfo.position = 999.0f;
    load.pixelInfo.albedo = color;
    load.pixelInfo.motionVector = 0.0f;
    load.pixelInfo.depth = generalBuffer.playerFar;
    load.pixelInfo.metallic  = 0.0f;
    load.pixelInfo.roughness = 0.0f;
    load.pixelInfo.viewDir = dir;
}