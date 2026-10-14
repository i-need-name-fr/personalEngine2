#include "../../include/ShaderStructures/ShaderStructure.h"

float raySphereIntersectNear(float3 rayOrigin , float3 rayDirection , float radius){
    float b = dot(rayOrigin , rayDirection);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    float disc = b * b - c;
    if(disc < 0.0f) return -1.0f;
    float sqrtDisc = sqrt(disc);
    float Near = -b - sqrtDisc;
    return (Near >= 0.0f) ? Near : 0.0f;
}

float RaySphereIntersectFar(float3 rayOrigin , float3 rayDirection , float radius){
    float b = dot(rayOrigin , rayDirection);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    float disc = b * b - c;
    if(disc < 0.0f) return -1.0f;
    float sqrtDisc = sqrt(disc);
    return -b + sqrtDisc;
}

float3 SampleExtincion(float h){
    float dRayL = exp(-h / H_RAYLEIGH);
    float dM = exp(-h / H_MIE);

    float3 rayL = BETA_RAYLEIGH_SCAT * dRayL;
    float3 mie = (BETA_MIE_ABS + BETA_MIE_SCAT) * dM;
    return rayL + mie;
}

[[vk::image_format("rgba16f")]]
[[vk::binding(0 , 0)]] RWTexture2D<float4> output;

[shader("compute")]
[numthreads(8 , 8 ,1)]
void main(uint3 id : SV_DispatchThreadID){
    if(id.x >= LUT_WIDTH || id.y >= LUT_HEIGHT) return;

    float2 uv = (float2(id.xy) + 0.5f) / float2(LUT_WIDTH , LUT_HEIGHT);
    float sunCosTheta = uv.x  * 2.0f - 1.0f;
    float3 sunDir = float3(sqrt(saturate(1.0f - sunCosTheta * sunCosTheta)),  0.0f , sunCosTheta);
    float altitude = lerp(PLANET_RADIUS , ATMOSPHERE_RADIUS, uv.y);
    float3 rayOrigin = float3(0.0f , 0.0f , altitude);

    float distToGround = raySphereIntersectNear(rayOrigin, sunDir , PLANET_RADIUS);
    if(distToGround > 0.0f){
        output[id.xy] = float4(0.0f , 0.0f , 0.0f , 1.0f);
        return;
    }

    float dist = RaySphereIntersectFar(rayOrigin , sunDir , ATMOSPHERE_RADIUS);
    float3 trans = float3(1.0f , 1.0f , 1.0f);

    float t = 0.0f;

    [unroll]
    for(uint i = 0 ; i < 40 ; i++){
        float newT = (float(i) + 0.3) / 40.0f * dist;
        float dt = newT - t;

        t = newT;

        float3 pos = rayOrigin + sunDir * t;
        float height = length(pos) - PLANET_RADIUS;

        float3 ext = SampleExtincion(height);

        trans *= exp(-dt * ext);
    }

    output[id.xy] = float4(trans , 1.0f);
}