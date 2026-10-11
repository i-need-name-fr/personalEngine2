struct VsOutput{
    float4 clipPos  : SV_Position;
    [[vk::location(0)]] float3 fragPos                    : TEXCOORD0;
    [[vk::location(1)]] float3 normal                     : TEXCOORD1;
    [[vk::location(2)]] float3x3 TBN                      : TEXCOORD2;
    [[vk::location(5)]] float2 uv                         : TEXCOORD5;
    [[vk::location(6)]] nointerpolation uint LUTindex     : TEXCOORD6;
};

#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::image_format("r32f")]]
[[vk::binding(0 , 0)]] RWTexture2D<float> pixelCheck;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2, 0)]] RWStructuredBuffer<InstanceData> instanceLayoutBuffer;
[[vk::binding(3 , 0)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(4 , 0)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(5 , 0)]] RWStructuredBuffer<LUT> LUT_Buffer;

[[vk::image_format("rgba16f")]]
[[vk::binding(6 , 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(7 , 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(8 , 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("r32f")]]
[[vk::binding(9 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("r32f")]]
[[vk::binding(10 , 0)]] RWTexture2D<float> roughnessImage;

[[vk::image_format("r32f")]]
[[vk::binding(11 , 0)]] RWTexture2D<float> metallicImage;

[[vk::binding(0 , 1)]] RWStructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState samp;
[[vk::binding(2 , 1)]] Texture2D allTextures[];


// true : the normal maps are OpenGL style ( green up ) , false : DirectX style ( green down )
static const bool NORMAL_MAP_OPENGL = true;

float2 calculateParallax(float2 currentUV , float3 V , int depthID){  
    if(depthID == NON_VALID_SLOT) return currentUV;
    const uint physicalIndex = virtualTextureIndex[depthID];
    const float depth = allTextures[physicalIndex].SampleLevel(samp , currentUV , 0).x;
    const float height = 1.0f - depth;
    const float height_scale = 0.001f;
    const float2 p = (V.xy) / V.z * height * height_scale;
    return currentUV - float2(p.x , -p.y);
}

[shader("pixel")]
void main(VsOutput input){

    uint2 pixel = uint2(input.clipPos.xy);

    // depth & position
    positionImage[pixel] = float4(input.fragPos, 1.0f);
    depthImage[pixel] = LinearizeDepth(input.clipPos.z , generalBuffer.playerNear , generalBuffer.playerFar);

    // current LUT 
    LUT lut = LUT_Buffer[input.LUTindex];

    const float3 V = normalize(generalBuffer.playerPosition - input.fragPos);
    const float3x3 TBN = input.TBN;
    const float3 V_TBN = mul(TBN , V);
    // parallax
    const float2 uv = calculateParallax(input.uv , V_TBN , lut.depthID);
    // normal  
    if(lut.normalID != NON_VALID_SLOT){
        const uint nID = virtualTextureIndex[lut.normalID];
        float3 n;
        n.xy = allTextures[nID].Sample(samp , uv).xy * 2.0f - 1.0f;
        n.z = sqrt(saturate(1.0f - dot(n.xy , n.xy)));
        n.y = NORMAL_MAP_OPENGL ? -n.y : n.y;
        n = normalize(mul(n , TBN));
        normalImage[pixel] = float4(n , 1.0f);
    }else{
        normalImage[pixel] = float4(input.normal , 1.0f);
    }

    if(lut.diffuseID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.diffuseID];
        albedoImage[pixel] = allTextures[id].Sample(samp , uv);
    }else{
        albedoImage[pixel] = float4(lut.color , 1.0f);
    }

    if(lut.metallicTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.metallicTextureID];
        metallicImage[pixel] = allTextures[id].Sample(samp , uv).x;
    }else{
        metallicImage[pixel] = lut.metallic;
    }

    if(lut.roughnessTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.roughnessTextureID];
        roughnessImage[pixel] = allTextures[id].Sample(samp , uv).x;
    }else{
        roughnessImage[pixel] = lut.roughness;
    }

    pixelCheck[uint2(input.clipPos.xy)] = COLOR_PIXEL;
}