struct VsOutput{
    float4 clipPos  : SV_Position;
    [[vk::location(0)]] float3 fragPos                    : TEXCOORD0;
    [[vk::location(1)]] float3 normal                     : TEXCOORD1;
    [[vk::location(2)]] float3x3 TBN                      : TEXCOORD2;
    [[vk::location(5)]] float2 uv                         : TEXCOORD5;
    [[vk::location(6)]] nointerpolation uint LUTindex     : TEXCOORD6;
};

#include "../../include/ShaderStructures/ShaderStructure.h"

// set 0 of the deferred render : only buffers , the g buffer is not here any more , it is written through the color attachments
[[vk::binding(0 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(4 , 0)]] StructuredBuffer<LUT> LUT_Buffer;

[[vk::binding(0 , 1)]] StructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState samp;
[[vk::binding(2 , 1)]] Texture2D allTextures[];

// the order of the SV_Target is the order of the color attachments in DeferredRender::executePass
struct PsOutput{
    float4 albedo     : SV_Target0;   // rgba16f
    float4 position   : SV_Target1;   // rgba32f
    float4 normal     : SV_Target2;   // rgba16f
    float  depth      : SV_Target3;   // r32f , linear
    float  roughness  : SV_Target4;   // r32f
    float  metallic   : SV_Target5;   // r32f
    float  pixelCheck : SV_Target6;   // r32f , COLOR_PIXEL where something was drawn
    float  objectAmbient : SV_Target7; // r32f , the ao map of the material ( 1 : there is no map ) , the direct light is multiplied by it
};

// the normal maps are OpenGL style ( green up ) or DirectX style ( green down ) : generalBuffer.normalMapOpenGL ( ImGui ) , the same flag as in RayTracing/GI_hit.hlsl

float2 calculateParallax(float2 currentUV , float3 V , int depthID){
    if(depthID == NON_VALID_SLOT) return currentUV;
    const uint physicalIndex = virtualTextureIndex[depthID];
    const float depth = allTextures[physicalIndex].SampleLevel(samp , currentUV , 0).x;
    const float height = 1.0f - depth;
    const float height_scale = 0.001f;
    const float2 p = (V.xy) / V.z * height * height_scale;
    return currentUV - float2(p.x , -p.y);
}

// the shader has no side effects and does not change the depth , so the depth test may run before it
[earlydepthstencil]
[shader("pixel")]
PsOutput main(VsOutput input){

    PsOutput output;

    // depth & position
    output.position = float4(input.fragPos, 1.0f);
    output.depth = LinearizeDepth(input.clipPos.z , generalBuffer.playerNear , generalBuffer.playerFar);

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
        // OpenGL maps ( green up ) are used as they are with this TBN , the DirectX ones ( green down ) are flipped . ( it was the other way round : every map was inverted )
        n.y = (generalBuffer.normalMapOpenGL) ? n.y : -n.y;
        n = normalize(mul(n , TBN));
        output.normal = float4(n , 1.0f);
    }else{
        output.normal = float4(input.normal , 1.0f);
    }

    if(lut.diffuseID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.diffuseID];
        output.albedo = allTextures[id].Sample(samp , uv);
    }else{
        output.albedo = float4(lut.color , 1.0f);
    }

    if(lut.metallicTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.metallicTextureID];
        // RGBA8 map : a grey picture has the value in every channel , an ARM map has the metallic in B
        output.metallic = allTextures[id].Sample(samp , uv).z;
    }else{
        output.metallic = lut.metallic;
    }

    if(lut.roughnessTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.roughnessTextureID];
        // RGBA8 map : a grey picture has the value in every channel , an ARM map has the roughness in G
        output.roughness = allTextures[id].Sample(samp , uv).y;
    }else{
        output.roughness = lut.roughness;
    }

    if(lut.armTextureID != NON_VALID_SLOT)
    {
        const uint id = virtualTextureIndex[lut.armTextureID];
        const float3 arm = allTextures[id].Sample(samp, uv).xyz;
        // will add occlusion later
        output.roughness = arm.y;
        output.metallic = arm.z;
    }


    // the ambient occlusion of the material ( objectAmbientImage ) : the value of the map ( the red channel ) or 1 when the material has no map . it is not the SSAO / HBAO :
    // the lighting multiplies only the DIRECT light by it ( not the ambient and not the indirect one )
    float ambientOcclusion = 1.0f;
    if(lut.aoTextureID != NON_VALID_SLOT)
    {
        const uint id = virtualTextureIndex[lut.aoTextureID];
        ambientOcclusion = allTextures[id].Sample(samp , uv).x;
    }
    output.objectAmbient = ambientOcclusion;

    // a perfect mirror ( roughness 0 ) makes the GGX degenerate : the lower limit is for every path above ( the value , the maps , the ARM )
    output.roughness = max(output.roughness , 0.1f);

    output.pixelCheck = COLOR_PIXEL;
    return output;
}
