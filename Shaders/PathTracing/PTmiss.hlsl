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

// the ao map of the material : 1 where there is no map . the DIRECT light is multiplied by it
[[vk::image_format("r32f")]]
[[vk::binding(39 , 0)]] RWTexture2D<float> objectAmbientImage;

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

[[vk::binding(34 ,0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(35 , 0)]] RWStructuredBuffer<uint> tileLightIndices;

// the G buffer of the LAST frame ( the images of the other parity ) : the same formats as the images of this frame
[[vk::image_format("rgba32f")]]
[[vk::binding(36 , 0)]] RWTexture2D<float4> prevPositionImage;

[[vk::image_format("rgba16")]]
[[vk::binding(37 , 0)]] RWTexture2D<float4> prevNormalImage;

[[vk::image_format("r32f")]]
[[vk::binding(38 , 0)]] RWTexture2D<float> prevDepthImage;

[[vk::binding(0 , 2)]] RWStructuredBuffer<lg::LightSource> lightBuffer;
[[vk::binding(1 , 2)]] RWStructuredBuffer<lg::LightCell> lightCells;
[[vk::binding(2 , 2)]] ConstantBuffer<lg::GridCascadeInfo> gridInfo;

// set 1 : the textures of the TextureManager2 ( the same set as in GI_hit.hlsl ) : the hit of a reflection ray reads its material from them
[[vk::binding(0 , 1)]] StructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState typicalSampler;
[[vk::binding(2 , 1)]] Texture2D allTextures[];

// the same payload as the closest hit of the primary bounce : the picture is written by the shaders themselves , the ray carries only the seed
struct [raypayload] RayPayLoad_primary_
{
    uint seed : write(caller , closesthit , miss) : read(caller , closesthit , miss);
};

typedef RayPayLoad_primary_ RayPayLoad;

uint bufferIndex(uint2 pixel){
    const uint2 clamped = min(pixel , uint2(generalBuffer.screenWidth - 1u , generalBuffer.screenHeight - 1u));
    return clamped.x + clamped.y * generalBuffer.screenWidth;
}

// ---------------------------------------------------------------- the sun disc ( the same as in LightPass and in RayTracing/generalMiss.hlsl )
static const float SUN_ANGULAR_RADIUS = 0.00465f;
static const float SUN_DISC_BRIGHTNESS = 100.0f;
static const float SUN_DISC_MAX = 60000.0f;

float3 sunTransmittanceToCamera(){
    const float u = skyBuffer.sunCosZenith * 0.5f + 0.5f;
    const float v = saturate(skyBuffer.cameraHeight / (ATMOSPHERE_RADIUS - PLANET_RADIUS));
    return transmissionLUT.SampleLevel(samp , float2(u , v) , 0).rgb;
}

float3 AddSunDisc(float3 skyColor , float3 rayDir){
    const float lightLength = length(sunBuffer.direction);
    // the light is the direction where the light goes : the sun is on the other side
    if(lightLength < 1e-5f) return skyColor;
    const float3 toSun = -sunBuffer.direction / lightLength;

    // nothing under the horizon
    if(rayDir.y < 0.0f) return skyColor;

    // the angle of one pixel from the projection ( projOnly[1][1] is 1 / tan( fov / 2 ) )
    const float pixelAngle = 2.0f / (max(generalBuffer.projOnly[1][1] , 1e-3f) * float(max(generalBuffer.screenHeight , 1u)));
    const float cosAngle = dot(rayDir , toSun);
    if(cosAngle < cos(SUN_ANGULAR_RADIUS + pixelAngle)) return skyColor;

    const float angle = acos(clamp(cosAngle , -1.0f , 1.0f));
    const float coverage = saturate((SUN_ANGULAR_RADIUS + 0.5f * pixelAngle - angle) / pixelAngle);

    const float3 disc = min(sunBuffer.diffuse * SUN_DISC_BRIGHTNESS * sunTransmittanceToCamera() , float3(SUN_DISC_MAX , SUN_DISC_MAX , SUN_DISC_MAX));
    return skyColor + disc * coverage;
}

// the motion of the sky : it has no position , only a direction ( w = 0 : the translation of the camera does not move it ) , so only the ROTATION of the camera counts .
// a zero vector here would give DLSS a trail of the sky when the camera turns
float2 skyMotionVector(const float3 dir , const uint2 pixel){
    const float2 size = float2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    const float2 uv = (float2(pixel) + 0.5f) / size;
    const float4 clip = mul(generalBuffer.prevViewProj , float4(dir , 0.0f));
    if(clip.w <= 1e-6f) return float2(0.0f , 0.0f);
    float2 prevUV = (clip.xy / clip.w) * 0.5f + 0.5f;
    prevUV.y = 1.0f - prevUV.y;
    return prevUV - uv;
}

// the primary ray left the scene : this pixel is the sky . the closest hit writes everything for a surface , so the miss writes everything for the sky ( the pixel check , the
// images that the denoisers and the next frame read , and an empty history of the reservoirs )
[shader("miss")]
void main(inout RayPayLoad load){
    const uint2 pixel = DispatchRaysIndex().xy;
    const float3 dir = WorldRayDirection();
    const float3 color = skybox.SampleLevel(samp , dir , 0).xyz;

    // the sky is far away : the biggest radius of the filter ( 10000 , the same as the miss of the ray tracing ) . PostProcess takes the sky from the diffuse image
    const float farDistance = 10000.0f;
    diffuseImage[pixel] = float4(AddSunDisc(color , dir) , farDistance);
    specularImage[pixel] = float4(0.0f , 0.0f , 0.0f , farDistance);
    indirectImage[pixel] = float4(0.0f , 0.0f , 0.0f , farDistance);

    pixelCheck[pixel] = SKYBOX_PIXEL;

    // w == 0 marks a sky texel ( checkDisocclusion and the denoisers drop the history of it )
    positionImage[pixel] = float4(999.0f , 999.0f , 999.0f , 0.0f);
    normalImage[pixel] = float4(-dir , 0.0f);
    albedoImage[pixel] = float4(color , 1.0f);
    viewImage[pixel] = float4(dir , 1.0f);
    depthImage[pixel] = generalBuffer.playerFar;
    roughnessImage[pixel] = 0.0f;
    metallicImage[pixel] = 0.0f;
    objectAmbientImage[pixel] = 1.0f;
    shadowImage[pixel] = float2(1.0f , 0.0f);
    motionVector[pixel] = skyMotionVector(dir , pixel);

    // otherwise the next frame would reuse whatever this slot held two frames ago
    ReservoirDI emptyDI;
    emptyDI.M = 0u;
    emptyDI.chosenIndex = 9999u;
    emptyDI.W = 0.0f;
    emptyDI.weightSum = 0.0f;
    emptyDI.diffuseColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
    emptyDI.specularColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
    ReSTIRDIBuffer[bufferIndex(pixel)] = emptyDI;
    ReSTIRGIBuffer[bufferIndex(pixel)] = (ReservoirGI)0;
}
