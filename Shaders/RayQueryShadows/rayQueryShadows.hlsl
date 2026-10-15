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


// now the traceRay

bool shadowTraceRay(float3 direction , float3 startPos , const float tMax){

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;

    RayDesc ray;
    ray.Origin = startPos;
    ray.Direction = direction;
    ray.TMax = tMax;
    ray.TMin = 0.01f;

    rayQuery.TraceRayInline(TLAS , RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES , 0xFF , ray);
    rayQuery.Proceed();
    return rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}


// the angular radius of the disc of the sun ( radians ) : the real one is 0.00465 . the rays go inside this cone around the direction to the sun ,
// so the shadow of a far object is soft and the one near the surface stays sharp
static const float SUN_ANGULAR_RADIUS = 0.00465f;

// a random direction inside of the cone around the axis ( uniform in the solid angle )
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

    // 0 samples : that source does not trace and does not shadow ( it is also no division by zero )

    // we will sample first the shadow from light
    float lightShadow = 1.0f;
    const uint lightSamples = features.RayQueryLightSampleCount;
    if(lightSamples > 0u){
        const uint lightID = ReSTIRDIBuffer[bufferIndex(id.xy , width , height)].chosenIndex;
        LightSource light = lightBuffer[lightID];

        float3 L = light.position - pos;
        const float dist = length(L);
        L = normalize(L);

        const float sinAlpha = (light.lightRadius / max(dist , light.lightRadius));
        const float cosAlpha = sqrt(1.0f - sinAlpha * sinAlpha);

        float visible = 0.0f;
        [loop]
        for(uint i = 0 ; i < lightSamples ; i++){
            uint seed = id.x * 832u + id.y * 293u  + generalBuffer.frameIndex * 2312u + i * 239u;
            const float3 dir = sampleCone(L , cosAlpha , seed);

            visible += (shadowTraceRay(dir , offset , dist)) ? 0.0f : 1.0f;
        }
        lightShadow = visible / float(lightSamples);
    }

    // now the sun shadow
    // we put negative sun direction , since we want to look to the sun , not from the sun
    float sunShadow = 1.0f;
    const uint sunSamples = features.RayQuerySunSampleCount;
    if(sunSamples > 0u){
        const float3 toSun = normalize(-sunBuffer.direction);
        const float sunCosAlpha = cos(SUN_ANGULAR_RADIUS);
        const float3 sunOrigin = pos + toSun * 0.01f;

        float visible = 0.0f;
        [loop]
        for(uint i = 0 ; i < sunSamples ; i++){
            // the other seed than the one of the lamp : the two sets of the rays must not repeat each other
            uint seed = id.x * 832u + id.y * 293u  + generalBuffer.frameIndex * 2312u + i * 239u + 15485863u;
            const float3 dir = sampleCone(toSun , sunCosAlpha , seed);

            visible += (shadowTraceRay(dir , sunOrigin , 1000.0f)) ? 0.0f : 1.0f;
        }
        sunShadow = visible / float(sunSamples);
    }

    const float rawShadow = lerp(lightShadow , sunShadow , features.light_sun_Proportion);
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