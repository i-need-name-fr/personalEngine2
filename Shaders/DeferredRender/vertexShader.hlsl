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

// set 0 of the deferred render : the bindings are the same in the fragment shader
// the vertex stage only reads them , so they are not writable here ( no vertexPipelineStoresAndAtomics needed )
[[vk::binding(0 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(1 , 0)]] StructuredBuffer<InstanceData> instanceLayoutBuffer;
[[vk::binding(2 , 0)]] StructuredBuffer<MeshData> meshBuffer;
[[vk::binding(3 , 0)]] StructuredBuffer<float4x4> instanceBuffer;


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
