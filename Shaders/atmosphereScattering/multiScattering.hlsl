#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] Texture2D transmissionLUT;
[[vk::binding(1 , 0)]] SamplerState samp;
[[vk::image_format("rgba16f")]]
[[vk::binding(2 , 0)]] RWTexture2D<float4> output;


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

// the directions of the sphere : sqrt of their count . the Mie phase ( g = 0.85 ) is a narrow lobe , 4 x 4 directions miss it
// ( 12% mean error , 60% at worst ) , 32 x 32 are 2% mean and 12% at worst . the LUT is computed once , so the count is cheap
static const uint SQRT_SAMPLES = 32;
static const uint DIR_SAMPLES  = SQRT_SAMPLES * SQRT_SAMPLES;
static const uint MS_STEPS     = 20; 
static const float GROUND_ALBEDO = 0.3;

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= LUT_WIDTH || DTid.y >= LUT_HEIGHT)
        return;

    float2 uv = (float2(DTid.xy) + 0.5) / float2(LUT_WIDTH, LUT_HEIGHT);

    float sunCosTheta = uv.x * 2.0 - 1.0;
    float3 sunDir = float3(sqrt(saturate(1.0 - sunCosTheta * sunCosTheta)), 0.0, sunCosTheta);

    float altitude = lerp(PLANET_RADIUS, ATMOSPHERE_RADIUS, uv.y);
    float3 rayOrigin = float3(0.0, 0.0, altitude);

    float3 Ltotal   = float3(0, 0, 0); 
    float3 LMStotal = float3(0, 0, 0); 

    for (uint i = 0; i < SQRT_SAMPLES; ++i)
    {
        for (uint j = 0; j < SQRT_SAMPLES; ++j)
        {
            float3 rayDir = StratifiedSphereDirection(i, j, SQRT_SAMPLES);

            float distToGround = RaySphereIntersectNear(rayOrigin, rayDir, PLANET_RADIUS);
            bool hitsGround = distToGround > 0.0;

            float tmax = hitsGround ? distToGround
                                     : RaySphereIntersectFar(rayOrigin, rayDir, ATMOSPHERE_RADIUS);

            float3 throughput = float3(1, 1, 1);
            float3 L   = float3(0, 0, 0);
            float3 LMS = float3(0, 0, 0);

            float cosThetaSun = dot(rayDir, sunDir);
            float phaseR = RayleighPhase(cosThetaSun);
            float phaseM = MiePhase(cosThetaSun, 0.85);

            // the midpoint rule over the whole ray : the end of every step was taken before , and the last part of the ray was lost
            const float dt = tmax / float(MS_STEPS);
            for (uint k = 0; k < MS_STEPS; ++k)
            {
                const float t = (float(k) + 0.5) * dt;

                float3 samplePos = rayOrigin + rayDir * t;
                float height = length(samplePos) - PLANET_RADIUS;

                float densityR, densityM;
                SampleDensities(height, densityR, densityM);

                float3 scatteringR = BETA_RAYLEIGH_SCAT * densityR;
                float  scatteringM = BETA_MIE_SCAT * densityM;
                float3 extinction  = scatteringR + scatteringM + BETA_MIE_ABS * densityM;

                float3 sampleTransmittance = exp(-dt * extinction);

                float sampleCosSun = dot(normalize(samplePos), sunDir);
                float3 sunTransmittance = SampleTransmittanceLUT(height, sampleCosSun);

                float3 Sint = (scatteringR * phaseR + scatteringM * phaseM) * sunTransmittance;
                L += throughput * (Sint - Sint * sampleTransmittance) / max(extinction, 1e-8);

                float3 scatteringTotal = scatteringR + scatteringM;
                float3 MSint = (scatteringTotal - scatteringTotal * sampleTransmittance) / max(extinction, 1e-8);
                LMS += throughput * MSint;

                throughput *= sampleTransmittance;
            }

            if (hitsGround)
            {
                float3 groundPos = rayOrigin + rayDir * tmax;
                float groundCosSun = dot(normalize(groundPos), sunDir);
                float3 transmittanceToGround = SampleTransmittanceLUT(0.0, groundCosSun);

                L += throughput * transmittanceToGround * saturate(groundCosSun) * (GROUND_ALBEDO / PI);
            }

            Ltotal   += L;
            LMStotal += LMS;
        }
    }

    Ltotal   /= float(DIR_SAMPLES);
    LMStotal /= float(DIR_SAMPLES);

    float3 result = Ltotal / max(1.0 - LMStotal, 1e-4);

    output[DTid.xy] = float4(result, 1.0);
}