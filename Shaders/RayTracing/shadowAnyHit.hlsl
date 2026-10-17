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


struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

[shader("anyhit")]
void main(inout ShadowRayPayLoad payload , in BuiltInTriangleIntersectionAttributes attr){
    const InstanceData instance = instanceLayout[InstanceID()];

    if(LUT_Buffer[instance.LUT_index].isLightSource != 0){
        IgnoreHit();
    }
}