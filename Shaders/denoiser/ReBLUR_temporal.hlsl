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

[[vk::image_format("rgba16f")]]
[[vk::binding(11, 0)]] RWTexture2D<float4> outputImage;

[[vk::image_format("r32f")]]
[[vk::binding(12, 0)]] RWTexture2D<float> pixelCheck;

static const float MAX_ACCUM_FRAMES = 32.0f;

#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(13 , 0)]] ConstantBuffer<GeneralFeatures> features;

bool checkDisocclusion(int3 currentUV , int3 prevUV){

    const float3 centerPosition = positionImage.Load(currentUV).xyz;
    const float3 centerNormal = normalImage.Load(currentUV).xyz;

    const float4 prevPositionSample = prevPositionImage.Load(prevUV);
    const float3 nNormal = prevNormalImage.Load(prevUV).xyz;   

    // w == 0 marks a sky texel written by the miss shader
    if(prevPositionSample.w == 0.0f) return false;

    const float positionDiff = length(centerPosition - prevPositionSample.xyz);
    if(positionDiff > 0.05f) return false;

    const float normalDot = dot(centerNormal , nNormal);
    if(normalDot < 0.9f) return false;
    
    return true;
}

[shader("compute")]
[numthreads(8, 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    inputImage.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    if(pixelCheck[id.xy] == SKYBOX_PIXEL){
        frameCount[id.xy] = 0.0f;
        return;
    }

    float2 uv = (float2(id.xy) + 0.5f ) / float2(screenSize);
    float2 prevUV = uv + motionVector[id.xy];
    // the pixel of the last frame : ROUNDED to the nearest one . the plain conversion to uint cut the number off ( always towards the top left ) , so every frame the history
    // moved half a pixel on average and a pixel that did not move at all ( mv = 1e-7 ) fell on its neighbour half of the time : the picture drifted and turned into soap
    const uint2 prevCoord = uint2(round(clamp(prevUV * float2(screenSize) - 0.5f , float2(0.0f , 0.0f) , float2(screenSize) - 1.0f)));

    float3 currentDiffuse = inputImage[id.xy].xyz;
    float pCount = prevFrameCount[prevCoord].x;
    float3 history = prevDiffuse[prevCoord].xyz;
    bool valid = !(any(prevUV > float2(1.0f , 1.0f)) || any(prevUV < float2(0.0f , 0.0f))) && checkDisocclusion(uint3(id.xy , 0) , uint3(prevCoord , 0));

    if(!valid){
        pCount = 0.0f;    
        history = currentDiffuse;
    }

    pCount = min(pCount + 1.0f , MAX_ACCUM_FRAMES);
    float alpha = 1.0f / pCount;
    float3 accumulated = lerp(history , currentDiffuse , alpha);
    frameCount[id.xy] = pCount;
    outputImage[id.xy] = float4(accumulated , 1.0f);
}