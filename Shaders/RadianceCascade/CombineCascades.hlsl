#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RWStructuredBuffer<RC::ProbeInfo> RCProbeBuffer;
[[vk::binding(1 , 0)]] RWStructuredBuffer<RC::CascadeInfo> RCCascadesBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<RC::GeneralProbeInfo> RCProbeInfo;
[[vk::image_format("rgba16f")]]
[[vk::binding(3 , 0)]] RWTexture2D<float4> RCImages[];

[[vk::push_constant]] struct 
{
    uint current;
    uint prev;
} push;


[shader("compute")]
[numthreads(8 ,8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const RC::CascadeInfo cas_ = RCCascadesBuffer[push.current];
    const uint2 cascadeSize = uint2(cas_.CASCADE_X , cas_.CASCADE_Y);
    // a casual check if we are out of range
    if(any(id.xy >= cascadeSize)) return;

    // now we need to get data from the image
    // but now the idea is simple
    // to get buildTexel 
    const float4 dataCombined_our = RCImages[push.current][id.xy];
    // the alpha is packed : the sign is " the ray hit something " , the rest is the distance of the hit ( RC::packHit )
    // if the ray hit something in its interval , we dont have to combine the data already , therefore goal is kinda simple
    bool hit_our;
    float distance_our;
    RC::unpackHit(dataCombined_our.a , hit_our , distance_our);
    if(hit_our) return;
    const uint2 probeCoord = (id.xy) / uint(cas_.BUILD_SIZE);
    const uint2 buildTexel = id.xy - (probeCoord * uint(cas_.BUILD_SIZE));
    // now we need to translate buildTexel into build UV , to get direction for that pixel
    const float2 buildUV = (float2(buildTexel) + 0.5f ) / float(cas_.BUILD_SIZE);
    const float3 dir = octahedralDecode(buildUV);

    const RC::CascadeInfo prevCas_ = RCCascadesBuffer[push.prev];
    const int prevBuildSize = int(prevCas_.BUILD_SIZE);

    const float2 currentScreenPixel = (float2(probeCoord) + 0.5f) * cas_.PROBE_RESOLUTION;
    const float2 prevProbePos = currentScreenPixel / prevCas_.PROBE_RESOLUTION - 0.5f;
    const int2 prevProbeBase = int2(floor(prevProbePos));
    const float2 prevProbeFrac = prevProbePos - float2(prevProbeBase);

    const float2 prevTexelPos = buildUV * float(prevBuildSize) - 0.5f;
    const int2 prevTexelBase = int2(floor(prevTexelPos));
    const float2 prevTexelFrac = prevTexelPos - float2(prevTexelBase);

    const int2 prevProbeCount = int2(prevCas_.CASCADE_X , prevCas_.CASCADE_Y) / prevBuildSize;

    const int PROBE_SAMPLE_COUNT = 1;
    const int BUILD_SAMPLE_COUNT = 1;

    // the samples of the previous cascade are unpacked one by one : the colour is the weighted mean , the hits are counted by their weights ( the coverage ) ,
    // the distance is the mean of the distances of the hits ( if there are not any : of the misses )
    float3 behindColor = 0.0f;
    float weightSum = 0.0f;
    float hitWeight = 0.0f;
    float hitDistanceSum = 0.0f;
    float missDistanceSum = 0.0f;

    [unroll]
    for(int probeY = 1 - PROBE_SAMPLE_COUNT ; probeY <= PROBE_SAMPLE_COUNT ; probeY++)
    {
        [unroll]
        for(int probeX = 1 - PROBE_SAMPLE_COUNT ; probeX <= PROBE_SAMPLE_COUNT ; probeX++)
        {
            // first we are going for probe
            const int2 currentProbeCoord = prevProbeBase + int2(probeX , probeY);
            if(any(currentProbeCoord < 0) || any(currentProbeCoord >= prevProbeCount)) continue;

            const uint probeIndex = prevCas_.PROBE_OFFSET + uint(currentProbeCoord.y) * prevCas_.PROBE_PER_ROW + uint(currentProbeCoord.x);
            if(RCProbeBuffer[probeIndex].validProbe == 0u) continue;

            const float2 probeDistance = abs(float2(probeX , probeY) - prevProbeFrac);
            const float probeWeight = saturate(1.0f - probeDistance.x / float(PROBE_SAMPLE_COUNT)) * saturate(1.0f - probeDistance.y / float(PROBE_SAMPLE_COUNT));
            if(probeWeight <= 0.0f) continue;

            [unroll]
            for(int buildY = 1 - BUILD_SAMPLE_COUNT ; buildY <= BUILD_SAMPLE_COUNT ; buildY++)
            {
                [unroll]
                for(int buildX = 1 - BUILD_SAMPLE_COUNT ; buildX <= BUILD_SAMPLE_COUNT ; buildX++)
                {
                    const int2 currentBuild = prevTexelBase + int2(buildX , buildY);
                    // since we dont want to touch other probes
                    if(any(currentBuild < 0) || any(currentBuild >= prevBuildSize)) continue;

                    const float2 texelDistance = abs(float2(buildX , buildY) - prevTexelFrac);
                    const float texelWeight = saturate(1.0f - texelDistance.x / float(BUILD_SAMPLE_COUNT)) * saturate(1.0f - texelDistance.y / float(BUILD_SAMPLE_COUNT));
                    if(texelWeight <= 0.0f) continue;

                    // the corner of the block of the probe + the texel in it
                    const int2 pixel = currentProbeCoord * prevBuildSize + currentBuild;

                    const float weight = probeWeight * texelWeight;
                    const float4 sampleData = RCImages[push.prev][uint2(pixel)];
                    bool sampleHit;
                    float sampleDistance;
                    RC::unpackHit(sampleData.a , sampleHit , sampleDistance);

                    behindColor += weight * sampleData.rgb;
                    weightSum += weight;
                    if(sampleHit)
                    {
                        hitWeight += weight;
                        hitDistanceSum += weight * sampleDistance;
                    }
                    else
                    {
                        missDistanceSum += weight * sampleDistance;
                    }
                }
            }
        }
    }

    // no probe of the previous cascade gave anything ( the edge of the screen , the probes in objects or on the sky ) : there is nothing to take , so the sample keeps what
    // it has . a ray that missed has the colour of the sky in rgb ( RCmiss ) , so the ambient is not lost , it is only not taken from the previous cascade
    if(weightSum <= 0.0f) return;

    // a ray that missed takes what is behind it : the colour ( the sky of a miss is not counted twice : our own colour does not count , the ray did not stop )
    // and the hit : it is a hit if the half of the weight is , the distance is the mean of the matching samples
    const bool behindHit = hitWeight * 2.0f >= weightSum;
    const float missWeight = max(weightSum - hitWeight , 1e-6f);
    const float behindDistance = behindHit ? (hitDistanceSum / max(hitWeight , 1e-6f)) : (missDistanceSum / missWeight);

    RCImages[push.current][id.xy] = float4(behindColor / weightSum , RC::packHit(behindHit , behindDistance));
}
