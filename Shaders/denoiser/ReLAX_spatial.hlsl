#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::image_format("rgba16f")]]
[[vk::binding(0, 0)]] RWTexture2D<float4> inputSpecular;

[[vk::image_format("rgba16f")]]
[[vk::binding(1, 0)]] RWTexture2D<float4> prevSpecular;

[[vk::image_format("r32f")]]
[[vk::binding(2, 0)]] RWTexture2D<float> frameCount;

[[vk::image_format("r32f")]]
[[vk::binding(3, 0)]] RWTexture2D<float> prevFrameCount;

[[vk::binding(4, 0)]] Texture2D depthImage;

[[vk::binding(5, 0)]] Texture2D prevDepthImage;

[[vk::binding(6, 0)]] Texture2D normalImage;

[[vk::binding(7, 0)]] Texture2D prevNormalImage;

[[vk::image_format("rg16f")]]
[[vk::binding(8, 0)]] RWTexture2D<float2> motionVector;

[[vk::binding(9, 0)]] Texture2D positionImage;

[[vk::binding(10, 0)]] Texture2D prevPositionImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(11, 0)]] RWTexture2D<float4> outputSpecular;

[[vk::image_format("r32f")]]
[[vk::binding(12, 0)]] RWTexture2D<float> pixelCheck;

[[vk::image_format("rgba32f")]]
[[vk::binding(13, 0)]] RWTexture2D<float4> viewDirImage;

[[vk::image_format("r32f")]]
[[vk::binding(14, 0)]] RWTexture2D<float> roughnessImage;


float NormalizeHitDistance(float hitDistance, float viewZ, float roughness) {
    float scale = lerp(viewZ, viewZ * roughness, roughness); // roughness=0 для ReBLUR можно опустить
    return hitDistance / max(scale, 1e-4);
}

static const float MAX_ACCUM_FRAMES = 32.0f;
static const float MAX_RADIUS_PIXELS = 32.0f;
static const float DEPTH_SIGMA = 8.0f;
static const float NORMAL_POWER = 128.0f;

#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(15 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(16 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;

static const float LOBE_VOLUME = 0.5f;

float ComputeBlurRadius(float roughness , float hitDistance , float viewDistance , float historyCount) {
    const float alpha = roughness * roughness;
    const float tanHalfAngle = alpha * LOBE_VOLUME / max(1.0f - LOBE_VOLUME , 1e-4f);
    const float pixelAngle = 2.0f / (max(generalBuffer.projOnly[1][1] , 1e-3f) * float(max(generalBuffer.screenHeight , 1u)));
    const float spotShare = hitDistance / max(viewDistance + hitDistance , 1e-4f);
    const float baseRadius = tanHalfAngle * spotShare / max(pixelAngle , 1e-6f);
    const float historyFactor = saturate(1.0 - historyCount / MAX_ACCUM_FRAMES);
    return min(baseRadius * lerp(0.25, 1.0, historyFactor) , MAX_RADIUS_PIXELS);
}

float normalPowerForRoughness(float roughness){
    return lerp(256.0f , 16.0 , roughness);
}

uint randomNumberIn(inout uint seed , const uint totalCount){
    return uint(rand(seed) * float(totalCount));
}

[shader("compute")]
[numthreads(8, 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    inputSpecular.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    if(pixelCheck[id.xy] == SKYBOX_PIXEL){
        return;
    }

    if(!generalBuffer.ReLAX){
        return;
    }

    float3 center = outputSpecular[id.xy].xyz;
    float historyCount = frameCount[id.xy].x;
    float hitDistance = inputSpecular[id.xy].a;

    if(hitDistance >= 999.0f){

        inputSpecular[id.xy]= outputSpecular[id.xy];
        return;
    }

    float3 centerNormal = normalImage[id.xy].xyz;
    float centerDepth = depthImage[id.xy].x;
    float roughness = roughnessImage[id.xy].x;

    // the distance from the camera to the surface along the ray ( not the depth along the axis ) : the mirror is this far from the virtual camera
    const float viewDistance = length(positionImage[id.xy].xyz - generalBuffer.playerPosition);
    const float radius = ComputeBlurRadius(roughness , hitDistance , viewDistance , historyCount);

    float3 sum = center;
    float weight= 1.0f;

    [unroll]
    for(uint i = 0 ; i < 32 ; i++){
        uint seed = id.x * 3942u  + id.y * 32849u + 239u * i;
        const uint random = randomNumberIn(seed , 16);
        float2 offset = PoissonDisk[random] * radius;
        int2 samplePos = int2(id.xy) + int2(offset);
        if(any(samplePos < int2(0 , 0)) || any(samplePos >= int2(screenSize))) continue;

        float sDepth = depthImage[samplePos].x;
        float3 sNormal =normalImage[samplePos].xyz;

        float depthW = exp(-abs(centerDepth - sDepth) /centerDepth * DEPTH_SIGMA);
        float normalW = pow(saturate(dot(sNormal , centerNormal)) , normalPowerForRoughness(roughness));

        float w = depthW * normalW;
        sum += outputSpecular[samplePos].xyz * w;
        weight += w;
    }

    inputSpecular[id.xy] = float4(sum / weight , 1.0f);
}