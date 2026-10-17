#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(2 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(3 , 0)]] RWStructuredBuffer<ReservoirDI> ReSTIRDIBuffer;
[[vk::binding(4 , 0)]] RWStructuredBuffer<LightSource> lightBuffer;

[[vk::binding(5 , 0)]] RaytracingAccelerationStructure TLAS;

[[vk::image_format("r32f")]]
[[vk::binding(6 , 0)]] RWTexture2D<float> shadowImage;

[[vk::image_format("r32f")]]
[[vk::binding(7 , 0)]] RWTexture2D<float> prevShadowImage;

[[vk::image_format("r32f")]]
[[vk::binding(8 , 0)]] RWTexture2D<float> pixelCheck;

[[vk::image_format("rgba32f")]]
[[vk::binding(9 , 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(10 , 0)]] RWTexture2D<float4> normalImage;

// what a hit is : TLAS gives the index of the instance ( InstanceID = instanceCustomIndex = the index in totalInstanceLayouts ) , the instance gives the material ( LUT ) ,
// and the material knows if the object is a body of a lamp
[[vk::binding(11 , 0)]] RWStructuredBuffer<InstanceData> instanceLayouts;
[[vk::binding(12 , 0)]] RWStructuredBuffer<LUT> luts;

// the lamps of the tile of the pixel ( Forward+ ) : the shadow is made from all of them , not from the one that ReSTIR DI has chosen
[[vk::binding(13 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(14 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(15 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;



bool shadowTraceRay(float3 direction , float3 startPos , const float tMax){

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE> rayQuery;

    RayDesc ray;
    ray.Origin = startPos;
    ray.Direction = direction;
    ray.TMax = tMax;
    ray.TMin = 0.01f;

    rayQuery.TraceRayInline(TLAS , RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES , 0xFF , ray);

    while(rayQuery.Proceed()){
        if(rayQuery.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;

        const InstanceData instance = instanceLayouts[rayQuery.CandidateInstanceID()];
        if(luts[instance.LUT_index].isLightSource != 0) continue;

        rayQuery.CommitNonOpaqueTriangleHit();
        return true;
    }

    return false;
}


static const float SUN_ANGULAR_RADIUS = 0.00465f;

static const float MAX_LIGHT_CONE_SIN = 0.05f;

float3 sampleCone(const float3 axis , const float cosAlpha , inout uint seed){
    const float phi = 2.0f * PI * rand(seed);
    const float rand2 = rand(seed);

    const float cosTheta = lerp(cosAlpha , 1.0f , rand2);
    const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    float3 tangent , bitangent;
    buildONB(axis , tangent , bitangent);
    return localDir.x * tangent + localDir.y * bitangent + localDir.z * axis;
}

float lightAttenuation(LightSource light , float lightDistance){
    const float falloff = 1.0f / (1.0f + light.lightLinear * lightDistance + light.lightQuadratic * lightDistance * lightDistance);
    const float ratio = saturate(lightDistance / max(light.lightRadius , 1e-4f));
    float window = saturate(1.0f - ratio * ratio * ratio * ratio);
    window *= window;
    return falloff * window;
}

void findLightIndices(uint2 currentPixel , out uint firstLight , out uint lightCount){
    const uint2 tileCoord = currentPixel / forwardInfo.TILE_SIZE;
    const uint tileIndex = tileCoord.x + tileCoord.y * forwardInfo.TILE_PER_ROW;

    lightCount = min(lightCountPerTile[tileIndex] , (uint)MAX_LIGHTS);
    firstLight = tileIndex * MAX_LIGHTS;
}

[shader("compute")]
[numthreads(8 , 8 ,1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    shadowImage.GetDimensions(width , height);
    uint2 screenSize = uint2(width , height);  
    if(any(id.xy >= screenSize)) return;
    if(pixelCheck[id.xy] == SKYBOX_PIXEL || pixelCheck[id.xy] == UNDEFINED_PIXEL){
        shadowImage[id.xy] = 1.0f;
        return;
    }

    if(features.PT || !features.RayQuery || features.VSM){
        shadowImage[id.xy] = 1.0f;
        return;
    }   

    const float3 pos = positionImage[id.xy].xyz;
    const float3 N = normalImage[id.xy].xyz;
    const float3 offset = pos + N * 0.0001f;

    float lightShadow = 1.0f;
    const uint lightSamples = features.RayQueryLightSampleCount;
    if(lightSamples > 0u){
        uint lightOffset , lightCount;
        findLightIndices(id.xy , lightOffset , lightCount);

        float weightSum = 0.0f;
        float visibleSum = 0.0f;

        [loop]
        for(uint l = 0 ; l < lightCount ; l++){
            const LightSource light = lightBuffer[tileLightIndices[lightOffset + l]];

            float3 L = light.position - pos;
            const float dist = length(L);
            L /= max(dist , 1e-4f);

            // the shadow of a lamp fades out beyond its radius ( lightShadowReach : 1 inside of the radius , 0 at lightRadius * ( 1 + lightShadowFade ) ) instead
            // of ending with a jump . the weight has no window of the radius any more : the window is in the reach
            const float reach = lightShadowReach(dist , light.lightRadius , features.lightShadowFade);
            const float falloff = 1.0f / (1.0f + light.lightLinear * dist + light.lightQuadratic * dist * dist);
            // the weight fades with the reach too : a lamp that ends its fade leaves the mean without a jump
            const float weight = luminance(light.diffuse) * falloff * saturate(dot(N , L)) * reach;
            if(reach <= 0.0f || weight <= 0.0f) continue;

            const float sinAlpha = min(light.lightRadius / max(dist , light.lightRadius) , MAX_LIGHT_CONE_SIN);
            const float cosAlpha = sqrt(1.0f - sinAlpha * sinAlpha);

            float visible = 0.0f;
            [loop]
            for(uint i = 0 ; i < lightSamples ; i++){
                uint seed = id.x * 832u + id.y * 293u  + generalBuffer.frameIndex * 2312u + i * 239u + l * 5683u;
                const float3 dir = sampleCone(L , cosAlpha , seed);

                visible += (shadowTraceRay(dir , offset , dist)) ? 0.0f : 1.0f;
            }
            visible /= float(lightSamples);
            // beyond the radius the shadow of this lamp is weaker and weaker : 1 ( no shadow ) at the end of the fade
            visible = lerp(1.0f , visible , reach);

            weightSum += weight;
            visibleSum += weight * visible;
        }

        if(weightSum > 0.0f) lightShadow = visibleSum / weightSum;
        lightShadow = pow(lightShadow , 2.0f);
    }

    float sunShadow = 1.0f;
    const uint sunSamples = features.RayQuerySunSampleCount;
    if(sunSamples > 0u){
        const float3 toSun = normalize(-sunBuffer.direction);
        const float sunCosAlpha = cos(SUN_ANGULAR_RADIUS);
        const float3 sunOrigin = pos + toSun * 0.01f;

        float visible = 0.0f;
        [loop]
        for(uint i = 0 ; i < sunSamples ; i++){
            uint seed = id.x * 832u + id.y * 293u  + generalBuffer.frameIndex * 2312u + i * 239u + 15485863u;
            const float3 dir = sampleCone(toSun , sunCosAlpha , seed);

            visible += (shadowTraceRay(dir , sunOrigin , 1000.0f)) ? 0.0f : 1.0f;
        }
        sunShadow = visible / float(sunSamples);
        
    }
    // will go like this , we dont lerp any more , since it wont work as i expected
    // now it will work like in теория вероятности , where we dont calculate straight the event , we find what is the chances of it not happening ( 1.0f - event )
    float nonShadow = 1.0f - (sunShadow + lightShadow);
    float rawShadow = 1.0f - nonShadow;
    float finalShadow;

    const float4 clipPos = mul(generalBuffer.prevViewProj , float4(pos,  1.0f));
    const float3 prevPos = clipPos.xyz / clipPos.w;
    float2 prevUV = prevPos.xy * 0.5f + 0.5f;
    prevUV.y = 1.0f - prevUV.y;

    bool validUV = !(any(prevUV > float2(1.0f , 1.0f)) || any(prevUV < float2(0.0f , 0.0f)));
    if(validUV){
        const uint2 prevPixel = uint2(  prevUV * float2(screenSize) -0.5f);
        const float prevShadow = prevShadowImage[prevPixel].x;
        finalShadow = lerp(rawShadow , prevShadow , features.temporalShadowK);
    }else{
        finalShadow = rawShadow;
    }

    shadowImage[id.xy]= finalShadow;
}