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

// DEFERRED RENDER WILL BE BEFORE RT FOR SPATIAL REUSE OF GI 
[shader("raygeneration")]
void main(){

    uint2 currentPixel = DispatchRaysIndex().xy;
    uint2 pixelSize = DispatchRaysDimensions().xy;
    if(pixelCheck[currentPixel] != COLOR_PIXEL){
        pixelCheck[currentPixel] = SKYBOX_PIXEL;
    }

    float2 uv = (float2(currentPixel) + 0.5f) / float2(pixelSize);
    float2 ndc = uv * 2.0 - 1.0f;
    ndc.y = -ndc.y;

    float4 pos = mul(generalBuffer.inverseProj , float4(ndc , 1.0f , 1.0f));
    float3 worldDir = mul((float3x3)generalBuffer.inverseView , pos.xyz).xyz;

    RayDesc ray;
    ray.Origin = generalBuffer.playerPosition;
    ray.Direction = normalize(worldDir);
    ray.TMax = 1000.0f;
    ray.TMin = 0.001f;

    RayPayLoad load = (RayPayLoad)0;
    load.seed = currentPixel.x * 4394u  + currentPixel.y * 9231u + generalBuffer.frameIndex *  230u;
    load.fromReflection = false;
    load.depth = 0;
    TraceRay(TLAS , RAY_FLAG_NONE , 0xFF , 0 , 0 , 0 , ray , load);
    diffuseImage[currentPixel.xy] = float4(load.lightInfo.diffuse , load.lightInfo.diffuseHitDistance);
    specularImage[currentPixel.xy] = float4(load.lightInfo.specular , load.lightInfo.specularHitDistance);
    indirectImage[currentPixel.xy] = float4(load.lightInfo.indirect , load.lightInfo.diffuseHitDistance);

    shadowImage[currentPixel.xy] = load.lightInfo.shadow;

    viewImage[currentPixel.xy] = float4(load.pixelInfo.viewDir , 1.0f);

    motionVector[currentPixel.xy] = load.pixelInfo.motionVector;
}