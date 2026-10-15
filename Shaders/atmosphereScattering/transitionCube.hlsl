#include "../../include/ShaderStructures/ShaderStructure.h"

// one descriptor set for both shaders of the AtmosphereScattering pipe ( SkyViewLUT uses 0 - 5 ) : the buffer is shared , the rest has its own numbers
[[vk::binding(0 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(6 , 0)]] Texture2D<float4> skyViewLUT;
[[vk::image_format("rgba16f")]]
[[vk::binding(7 , 0)]] RWTexture2DArray<float4> output; // 6 граней как слои массива
// the azimuth wraps around ( repeat in U ) , SkyViewLUT has its own sampler with the clamp
[[vk::binding(8 , 0)]] SamplerState linearSampler;

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

    float u = frac(realAzimuth / (2.0 * PI) + 1.0);
    float l = sign(elevation) * sqrt(saturate(abs(elevation) / (PI * 0.5)));
    float v = l * 0.5 + 0.5;

    output[uint3(DTid.xy, face)] = skyViewLUT.SampleLevel(linearSampler, float2(u, v), 0);
}