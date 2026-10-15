[[vk::image_format("rgba16f")]]
[[vk::binding(0, 0)]] RWTexture2D<float4> inputImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(1, 0)]] RWTexture2D<float4> prevDiffuse;

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

// WE WERE WRITING IN THIS , NOW WE ARE WRITING BACK TO INPUT
[[vk::image_format("rgba16f")]]
[[vk::binding(11, 0)]] RWTexture2D<float4> outputImage;

[[vk::image_format("r32f")]]
[[vk::binding(12, 0)]] RWTexture2D<float> pixelCheck;

float NormalizeHitDistance(float hitDistance, float viewZ, float roughness) {
    // упрощённая версия формулы NRD — есть более точная с параметрами,
    // но идея та же: делим на масштаб сцены в этой точке
    float scale = lerp(viewZ, viewZ * roughness, roughness); // roughness=0 для ReBLUR можно опустить
    return hitDistance / max(scale, 1e-4);
}

static const float MAX_ACCUM_FRAMES = 32.0f;
static const float MAX_RADIUS_PIXELS = 32.0f;
static const float DEPTH_SIGMA = 8.0f;
static const float NORMAL_POWER = 128.0f;

#include "../../include/ShaderStructures/ShaderStructure.h"
[[vk::binding(13 , 0)]] ConstantBuffer<GeneralFeatures> features;


float ComputeBlurRadius(float normalizedHitDist, float historyCount) {
    float baseRadius = normalizedHitDist * MAX_RADIUS_PIXELS;
    // чем длиннее история — тем меньше нужен spatial-радиус (твой предыдущий вопрос)
    float historyFactor = saturate(1.0 - historyCount / MAX_ACCUM_FRAMES);
    return baseRadius * lerp(0.25, 1.0, historyFactor); // не даём радиусу упасть в 0 совсем
}

[shader("compute")]
[numthreads(8, 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    inputImage.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    if(pixelCheck[id.xy] == SKYBOX_PIXEL){
        return;
    }

    float3 center = outputImage[id.xy].xyz;
    float historyCount = frameCount[id.xy].x;
    float hitDistance = inputImage[id.xy].a;

    if(hitDistance >= 999.0f){

        inputImage[id.xy]= outputImage[id.xy];
        return;
    }

    float radius = ComputeBlurRadius(NormalizeHitDistance(hitDistance , depthImage[id.xy].x , 0) ,historyCount );
    
    float3 centerNormal = normalImage[id.xy].xyz;
    float centerDepth = depthImage[id.xy].x;

    float3 sum = center;
    float weight= 1.0f;

    [unroll]
    for(uint i = 0 ; i < 16 ; i++){
        float2 offset = PoissonDisk[i] * radius;
        int2 samplePos = int2(id.xy) + int2(offset);
        if(any(samplePos < int2(0 , 0)) || any(samplePos >= int2(screenSize))) continue;

        float sDepth = depthImage[samplePos].x;
        float3 sNormal =normalImage[samplePos].xyz;

        float depthW = exp(-abs(centerDepth - sDepth) /centerDepth * DEPTH_SIGMA);
        float normalW = pow(saturate(dot(sNormal , centerNormal)) , NORMAL_POWER);

        float w = depthW * normalW;
        // neighbours come from temporal's output (outputDiffuse) , never from inputDiffuse -
        // other invocations of this dispatch are writing inputDiffuse right now
        sum += outputImage[samplePos].xyz * w;
        weight += w;
    }

    inputImage[id.xy] = float4(sum / weight , 1.0f);
}