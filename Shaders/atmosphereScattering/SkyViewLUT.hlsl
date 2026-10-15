#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(1 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(2 , 0)]] Texture2D<float4> transmissionLUT;
[[vk::binding(3 , 0)]] Texture2D<float4> multiScatter;
[[vk::binding(4 , 0)]] SamplerState samp;

[[vk::image_format("rgba16f")]]
[[vk::binding(5 , 0)]] RWTexture2D<float4> output;


float RaySphereIntersectNear(float3 ro, float3 rd, float r)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float sqrtDisc = sqrt(disc);
    float tNear = -b - sqrtDisc;
    return (tNear >= 0.0) ? tNear : -1.0;
}

float RaySphereIntersectFar(float3 ro, float3 rd, float r)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float sqrtDisc = sqrt(disc);
    return -b + sqrtDisc;
}

void SampleDensities(float height, out float densityRayleigh, out float densityMie)
{
    densityRayleigh = exp(-height / H_RAYLEIGH);
    densityMie      = exp(-height / H_MIE);
}

float3 SampleTransmittanceLUT(float height, float cosSunZenith)
{
    float u = cosSunZenith * 0.5 + 0.5;
    float v = saturate(height / (ATMOSPHERE_RADIUS - PLANET_RADIUS));
    return transmissionLUT.SampleLevel(samp, float2(u, v), 0).rgb;
}

float3 SampleMultiScatter(float height, float cosSunZenith)
{
    float u = cosSunZenith * 0.5 + 0.5;
    float v = saturate(height / (ATMOSPHERE_RADIUS - PLANET_RADIUS));
    return multiScatter.SampleLevel(samp, float2(u, v), 0).rgb;
}


float3 StratifiedSphereDirection(uint i, uint j, uint sqrtSamples) 
{
    float u = (float(i) + 0.5) / float(sqrtSamples);
    float v = (float(j) + 0.5) / float(sqrtSamples);

    float cosTheta = 1.0 - 2.0 * u; 
    float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    float phi = 2.0 * PI * v;

    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float RayleighPhase(float cosTheta)
{
    return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
}

float MiePhase(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 1e-4), 1.5));
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    if(id.x >= LUT_WIDTH || id.y >= LUT_HEIGHT) return;

    float2 uv = (float2(id.xy) + 0.5f) / float2(LUT_WIDTH , LUT_HEIGHT);

    float azimuth = uv.x * 2.0f * PI;
    float l = uv.y * 2.0f - 1.0f;
    float elevation = l * l * sign(l) * PI * 0.5f;

    float3 rayDir = float3(cos(elevation) * cos(azimuth) , sin(azimuth) * cos(elevation) , sin(elevation));

    float sinSunZenith = sqrt(saturate(1.0f - skyBuffer.sunCosZenith * skyBuffer.sunCosZenith));
    float3 sunDir = float3(sinSunZenith, 0.0f , skyBuffer.sunCosZenith);

    float earthCenterDist = PLANET_RADIUS + skyBuffer.cameraHeight;
    float3 rayOrigin = float3(0.0f , 0.0f , earthCenterDist);

    float disToGround = RaySphereIntersectNear(rayOrigin , rayDir , PLANET_RADIUS);
    float tmax = (disToGround > 0.0f ) ? disToGround : RaySphereIntersectFar(rayOrigin , rayDir , ATMOSPHERE_RADIUS);

    if(tmax <= 0.0f){
        output[id.xy] = float4(0.0f , 0.0f , 0.0f , 1.0f);
        return;
    }

    float cosThetaSun = dot(rayDir , sunDir);
    float phaseR = RayleighPhase(cosThetaSun);
    float phaseM = MiePhase(cosThetaSun , 0.85f);

    float3 throughput = float3(1 , 1 , 1);
    float3 L = float3(0 , 0 , 0);
    // the midpoint rule over the whole ray ( t is a distance in meters : the old code divided by tmax , so t was ~1e-7 and the sky was black )
    const float dt = tmax / float(SKY_VIEW_STEPS);

    [loop]
    for(uint i = 0 ; i < SKY_VIEW_STEPS ; i++){

        const float t = (float(i) + 0.5f) * dt;

        float3 samplePos = rayOrigin + rayDir * t;
        float height = length(samplePos) - PLANET_RADIUS;

        float densityR , densityM;
        SampleDensities(height , densityR , densityM);

        float3 scatteringR = BETA_RAYLEIGH_SCAT * densityR;
        float scatteringM = BETA_MIE_SCAT * densityM;
        float3 ext = scatteringR + scatteringM +  BETA_MIE_ABS * densityM;

        float3 sampleTrans = exp(-dt * ext);

        float sampleCosSun = dot(normalize(samplePos) , sunDir);
        float3 sunTrans = SampleTransmittanceLUT(height , sampleCosSun);
        float3 multi = SampleMultiScatter(height , sampleCosSun);

        float3 single = (scatteringR * phaseR + scatteringM * phaseM) * sunTrans;
        float3 S = single + ( scatteringR + scatteringM) * multi;
        float3 Sint = (S - S * sampleTrans) / max(ext , 1e-8);
        L += throughput * Sint;
        throughput *= sampleTrans;
    }

    output[id.xy] = float4(L , 1.0f);
}