#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;

[[vk::image_format("rgba16")]]
[[vk::binding(1 , 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(2 , 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("r32f")]]
[[vk::binding(3 , 0)]] RWTexture2D<float> pixelCheck;

[[vk::binding(4 , 0)]] RWStructuredBuffer<RC::ProbeInfo> RCProbeBuffer;
[[vk::binding(5 , 0)]] RWStructuredBuffer<RC::CascadeInfo> RCCascadesBuffer;

[[vk::push_constant]] struct {
    uint id;
} push;

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const RC::CascadeInfo cas_ = RCCascadesBuffer[push.id];

    // the number of the probes on the axes : the image of the cascade is probes * BUILD_SIZE
    const uint2 probeCount = uint2(cas_.CASCADE_X , cas_.CASCADE_Y) / cas_.BUILD_SIZE;
    if(any(id.xy >= probeCount)) return;

    const uint probeIndex = cas_.PROBE_OFFSET + id.y * cas_.PROBE_PER_ROW + id.x;

    // the centre of the tile , the last tile is cut by the edge of the screen : the pixel stays inside of the screen
    const int2 screenSize = int2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    const int2 pixel = min(int2((float2(id.xy) + 0.5f) * cas_.PROBE_RESOLUTION) , screenSize - 1);

    RCProbeBuffer[probeIndex].cascadeID = push.id;

    // not a surface ( the sky ) : no probe here
    if(pixelCheck[pixel] != COLOR_PIXEL)
    {
        RCProbeBuffer[probeIndex].validProbe = 0u;
        return;
    }

    const float3 position = positionImage[pixel].xyz;
    RCProbeBuffer[probeIndex].position = position;
    RCProbeBuffer[probeIndex].normal = normalize(normalImage[pixel].xyz);
    RCProbeBuffer[probeIndex].cameraDistance = length(position - generalBuffer.playerPosition);
    RCProbeBuffer[probeIndex].validProbe = 1u;
}
