#include "../../include/ShaderStructures/ShaderStructure.h"

// one descriptor set for both shaders of the AtmosphereScattering pipe ( SkyViewLUT uses 0 - 5 ) : the buffer is shared , the rest has its own numbers
[[vk::binding(0 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(6 , 0)]] Texture2D<float4> skyViewLUT;
[[vk::image_format("rgba16f")]]
[[vk::binding(7 , 0)]] RWTexture2DArray<float4> output; // 6 граней как слои массива
// the azimuth wraps around ( repeat in U ) , SkyViewLUT has its own sampler with the clamp
[[vk::binding(8 , 0)]] SamplerState linearSampler;

// the albedo of the ground that the mirrored sky is multiplied by under the horizon ( a real ground reflects 10 - 30 % , not 100 % ) , and the angle under
// the horizon where that change is made ( 0.1 rad ~ 6 degrees )
static const float3 GROUND_COLOR = float3(0.30f , 0.28f , 0.20f);
static const float GROUND_FADE_RANGE = 0.1f;

// Стандартная конвертация (face, uv [0,1]) -> направление в мировых координатах
float3 GetCubemapDirection(uint face, float2 uv)
{
    float2 ndc = uv * 2.0 - 1.0; // [-1,1]
    float3 dir;
    switch (face)
    {
        case 0: dir = float3( 1.0, -ndc.y, -ndc.x); break; // +X
        case 1: dir = float3(-1.0, -ndc.y,  ndc.x); break; // -X
        case 2: dir = float3( ndc.x,  1.0,  ndc.y); break; // +Y
        case 3: dir = float3( ndc.x, -1.0, -ndc.y); break; // -Y
        case 4: dir = float3( ndc.x, -ndc.y,  1.0); break; // +Z
        default: dir = float3(-ndc.x, -ndc.y, -1.0); break; // -Z
    }
    return normalize(dir);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint face = DTid.z;
    if (DTid.x >= FACE_SIZE || DTid.y >= FACE_SIZE || face >= 6)
        return;

    float2 uv = (float2(DTid.xy) + 0.5) / float(FACE_SIZE);
    float3 dir = GetCubemapDirection(face, uv);

    // the cube is in the world : Y is up ( the LUT is Z up ) , so the horizontal plane is X Z and the height is Y
    // the sun is at the azimuth 0 in the LUT : the cube is turned by the azimuth of the sun in the world
    const float sunAzimuth = atan2(skyBuffer.sunSinAzimuth , skyBuffer.sunCosAzimuth);
    float realAzimuth = atan2(dir.z, dir.x) - sunAzimuth;
    float elevation = asin(clamp(dir.y , -1.0 , 1.0));

    // below the horizon there is the ground , not the sky ( the LUT ray ends in the ground there and brings almost nothing : the lower half of the cube was black ,
    // and that is what the ambient light of the normals that look down and the far edge of the world saw ) . the ground is the mirrored sky times the colour of
    // the ground : at night the sky is dark and so is the ground , the horizon has no seam ( the mirror continues the sky ) , and the colour fades in over
    // GROUND_FADE_RANGE radians under the horizon , so there is no hard line either
    const float below = saturate(-elevation / GROUND_FADE_RANGE);
    const float groundBlend = below * below * (3.0 - 2.0 * below);
    elevation = abs(elevation);

    float u = frac(realAzimuth / (2.0 * PI) + 1.0);
    float l = sqrt(saturate(elevation / (PI * 0.5)));
    float v = l * 0.5 + 0.5;

    const float4 sky = skyViewLUT.SampleLevel(linearSampler, float2(u, v), 0);
    output[uint3(DTid.xy, face)] = float4(sky.rgb * lerp(float3(1.0 , 1.0 , 1.0) , GROUND_COLOR , groundBlend) , sky.a);
}