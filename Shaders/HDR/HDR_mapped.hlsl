#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RWStructuredBuffer<uint> history;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<GeneralFeatures> features;

[[vk::image_format("rgba16f")]]
[[vk::binding(3 , 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(4 , 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(5 , 0)]] RWTexture2D<float4> HDRImage;

[[vk::image_format("r32f")]]
[[vk::binding(6 , 0)]] RWTexture2D<float> pixelCheck;

[[vk::image_format("r32f")]]
[[vk::binding(7 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("r32f")]]
[[vk::binding(8 , 0)]] RWTexture2D<float> roughnessImage;

[[vk::image_format("r32f")]]
[[vk::binding(9 , 0)]] RWTexture2D<float> metallicImage;

[[vk::image_format("rgba8")]]
[[vk::binding(10 , 0)]] RWTexture2D<float4> finalImage;

// the AO after the bilateral filter : only the AO_MODE shows it
[[vk::image_format("r32f")]]
[[vk::binding(11 , 0)]] RWTexture2D<float> filteredAOImage;

// the shadow after the bilateral filter : only the SHADOW_MODE shows it
[[vk::image_format("r32f")]]
[[vk::binding(12 , 0)]] RWTexture2D<float> filteredShadowImage;


float3 ACESFilm(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

static float minLogLum = -10.0f;
static float logLumRange = 12.0f;

// the average used to be computed by EVERY pixel : 255 reads of the histogram each , ~900 million reads
// per frame at 2560x1440 ( the pass took ~430 ms ) . now the 64 threads of a group split the bins between
// them and add the partial sums up in shared memory - 64 times fewer reads , the same result
groupshared float groupWeighted[64];
groupshared uint groupCount[64];

// has to be called by EVERY thread of the group ( it has barriers ) , so nobody may return before it
float ComputeAverageLuminance(RWStructuredBuffer<uint> histogram , uint localIndex)
{
    float weightedSum = 0.0;
    uint countedPixels = 0;

    // bin 0 is skipped ( almost black , it must not pull the average down ) , thread t takes t , t + 64 , t + 128 , t + 192
    for (uint i = localIndex; i < 256; i += 64)
    {
        if(i == 0) continue;
        float logLum = minLogLum + (float(i - 1) / 254.0) * logLumRange;
        weightedSum += histogram[i] * logLum;
        countedPixels += histogram[i];
    }

    groupWeighted[localIndex] = weightedSum;
    groupCount[localIndex] = countedPixels;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if(localIndex < stride){
            groupWeighted[localIndex] += groupWeighted[localIndex + stride];
            groupCount[localIndex] += groupCount[localIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // nothing to average ( the whole image is darker than bin 1 ) : answer with the darkest brightness the histogram knows ,
    // so that the exposure goes to its maximum instead of the 1.0 that an empty sum would give
    if(groupCount[0] == 0) return exp2(minLogLum);

    float avgLogLum = groupWeighted[0] / groupCount[0];
    return exp2(avgLogLum);   // обратно из log-пространства в реальную яркость
}

[shader("compute")]
[numthreads(8, 8 ,1)]
void main(uint3 id : SV_DispatchThreadID , uint localIndex : SV_GroupIndex){
    uint width , height;
    HDRImage.GetDimensions(width , height);

    // the average first : all the threads of the group take part in it , the ones outside of the image too
    const float averageLuminance = ComputeAverageLuminance(history , localIndex);

    if(id.x >= width || id.y >= height) return;

    // a debug view ( albedo , normal , depth , indirect ) : PostProcess already wrote the colour to show , exposure and the tone mapping
    // would only change it . ( after the average above , the barriers in it need every thread of the group )

    if(features.currentMode == ALBEDO_MODE){
        // albedo image is surely rgba8 , since its diretly from textures
        finalImage[id.xy] = float4( albedoImage[id.xy].xyz, 1.0f);
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == NORMAL_MODE){
        float3 normal = normalImage[id.xy].xyz;
        normal = normal * 0.5f + 0.5f;
        finalImage[id.xy] = float4( normal, 1.0f);
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == METALLIC_MODE){
        const float metallic = metallicImage[id.xy].x;
        finalImage[id.xy] = float4( metallic , metallic , metallic , 1.0f);
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == ROUGHNESS_MODE){
        const float rough = roughnessImage[id.xy].x;
        finalImage[id.xy] = float4(rough , rough , rough , 1.0f);
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == AO_MODE){
        const float ao = filteredAOImage[id.xy].x;
        finalImage[id.xy] = float4(ao , ao , ao , 1.0f);
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == SHADOW_MODE){
        // the scalar as it is in the image ( PostProcess takes it to the power of shadow thickness after it ) . 1 = lit , 0 = black . more than 1 is
        // a bug of the formula ( a sum of two shadows ) : it is red
        const float shadow = filteredShadowImage[id.xy].x;
        finalImage[id.xy] = (shadow > 1.001f) ? float4(1.0f , 0.0f , 0.0f , 1.0f) : float4(shadow , shadow , shadow , 1.0f);
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }

    const float gamma = 2.2f;
    const float averageExposure =  0.18f / averageLuminance;
    const float exposure = clamp(averageExposure , 0.05f , 6.0f) * max(generalBuffer.exposure , 0.0f);
    const float3 hdr = HDRImage[id.xy].xyz;
    const float3 finalColor = hdr * exposure;
    const float3 toneMapped = ACESFilm(finalColor);
    const float3 result = pow(toneMapped , 1.0f/gamma);
    finalImage[id.xy] = float4(result , 1.0f);
    pixelCheck[id.xy] = UNDEFINED_PIXEL;
}