#include "../../include/ShaderStructures/ShaderStructure.h"

struct VS_INPUT{
    // general vertex
    [[vk::location(0)]] float3 position  : POSITION;
    [[vk::location(1)]] float3 normal    : NORMAL;
    [[vk::location(2)]] float3 tangent   : TANGENT;
    [[vk::location(3)]] float3 bitangent : BITANGENT;
    [[vk::location(4)]] float2 uv        : TEXCOORD0;

};

struct VS_OUTPUT{
    float4 clipPos  : SV_Position;
    [[vk::location(0)]] float3 fragPos                    : TEXCOORD0;
    [[vk::location(1)]] float3 normal                     : TEXCOORD1;
    [[vk::location(2)]] float3x3 TBN                      : TEXCOORD2;
    [[vk::location(5)]] float2 uv                         : TEXCOORD5;
    [[vk::location(6)]] nointerpolation uint LUTindex     : TEXCOORD6;
};

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


[shader("vertex")]
VS_OUTPUT main(VS_INPUT input , uint instanceID : SV_InstanceID){
    VS_OUTPUT output;

    InstanceData currentInstance = instanceLayoutBuffer[instanceID];
    uint meshID = currentInstance.meshDataIndex;
    uint instanceBufferIndex = meshBuffer[meshID].instanceIndex;
    float4x4 model = instanceBuffer[instanceBufferIndex];
    uint LUT_index = currentInstance.LUT_index;

    output.fragPos = mul(model , float4(input.position , 1.0f)).xyz;
    // normalMatrix3x3 builds its result from cross products of m's ROWS, which already
    // gives inverse(transpose(m)) directly (verified: cross(r1,r2),cross(r2,r0),cross(r0,r1)
    // over det == inverse(transpose(m)) exactly, since inverse(N)'s rows are pairwise
    // cross products of N's COLUMNS, and here those "columns" being crossed are m's rows,
    // i.e. N = transpose(m)). so the transpose is already baked in - putting the vector on
    // the left here re-transposed it back down to plain inverse(model), same mistake as
    // the tangent/bitangent w=1 bug: matrix must be on the left, same as for position.
    output.normal = normalize(mul(normalMatrix3x3((float3x3)model) , input.normal));
    output.LUTindex = LUT_index;
    output.uv = input.uv;
    const float3 T = normalize(mul(model , float4(input.tangent , 0.0f)).xyz);
    const float3 B = normalize(mul(model , float4(input.bitangent , 0.0f)).xyz);
    output.TBN = float3x3(T , B , output.normal);
    output.clipPos = mul(generalBuffer.viewProj , float4(output.fragPos , 1.0f));
    return output;
}