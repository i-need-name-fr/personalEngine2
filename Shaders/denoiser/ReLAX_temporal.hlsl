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

// temp buffer // gonna add a typical buffer from push constant
[[vk::binding(15 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(16 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;

static const float MAX_ACCUM_FRAMES = 32.0f;

float3 ReconstructWorldPos(float2 uv , float depth , float4x4 invViewProj){
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    float4 ndcPos = float4(ndc , depth , 1.0f);
    float4 ndcWorld = mul(invViewProj , ndcPos);
    return ndcWorld.xyz / ndcWorld.w;
}

float3 ComputeVirtualPosition(float3 worldPos, float3 normal, float3 viewDir, float hitDistance) {
    float3 reflectDir = reflect(-viewDir, normal);
    return worldPos + reflectDir * hitDistance; // "куда бы попал взгляд, если бы прошёл сквозь отражение"
}

float2 ComputeVirtualMotionVector(float3 currentVirtualPos, float4x4 prevViewProj, float2 currentUV) {
    float4 prevClip = mul(prevViewProj, float4(currentVirtualPos, 1.0));
    // behind the previous camera : outside of the screen , the caller drops the history
    if(prevClip.w <= 0.0f) return currentUV - float2(-2.0f , -2.0f);

    float2 prevUV = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
    // the same flip as the other passes ( ndc y goes up , the uv goes down )
    prevUV.y = 1.0f - prevUV.y;
    return currentUV - prevUV;
}

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
[numthreads(8 ,8 ,1)]
void main(uint3 id : SV_DispatchThreadID){
    uint width , height;
    inputSpecular.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    if(pixelCheck[id.xy] == SKYBOX_PIXEL ){
        frameCount[id.xy] = 0.0f;
        return;
    }

    float2 uv = (float2(id.xy) + 0.5f ) / float2(screenSize);
    float3 current = inputSpecular[id.xy].xyz;
    float roughness = roughnessImage[id.xy].x;
    float3 normal = normalImage[id.xy].xyz;
    float3 V = viewDirImage[id.xy].xyz;

    // depthImage holds real linearized depth, not NDC z - ReconstructWorldPos expects NDC z and
    // would produce garbage here (same issue fixed earlier in AmbientOcclusion.hlsl). positionImage
    // is already bound and holds the world position directly, so just read it.
    float3 P = positionImage[id.xy].xyz;
    float hitDistance = inputSpecular[id.xy].a;

    float3 virtualPos = ComputeVirtualPosition(P , normal , V ,hitDistance );
    float2 virtualUV = ComputeVirtualMotionVector(virtualPos , generalBuffer.prevViewProj , uv);
    float2 prevUV = uv - virtualUV;

    const uint2 prevCoord = uint2(clamp(prevUV * float2(screenSize) - 0.5f , float2(0.0f , 0.0f) , float2(screenSize) - 1.0f));

    // the history is read only when it is on the screen
    bool valid = !(any(prevUV > float2(1.0f , 1.0f)) || any(prevUV < float2(0.0f , 0.0f)));
    valid = valid && checkDisocclusion(uint3(id.xy , 0) , uint3(prevCoord , 0));

    float pCount = 0.0f;
    float3 history = current;

    if(valid){
        pCount = prevFrameCount[prevCoord].x;
        history = prevSpecular[prevCoord].xyz;
    }

    pCount = min(pCount + 1.0f , MAX_ACCUM_FRAMES);
    float alpha = 1.0f / pCount;
    float3 accumulated = lerp(history , current , alpha);
    frameCount[id.xy] = pCount;
    outputSpecular[id.xy] = float4(accumulated , 1.0f);
}