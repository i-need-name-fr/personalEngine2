#include "../../include/ShaderStructures/ShaderStructure.h"


[[vk::image_format("r32f")]]
[[vk::binding(0 , 0)]] RWTexture2D<float> src;

[[vk::image_format("r32f")]]
[[vk::binding(1 , 0)]] RWTexture2D<float> dst;

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width , height;
    dst.GetDimensions(width , height);
    const uint2 size = uint2(width , height);
    if(any(id.xy >= size)) return;

    const uint2 srcPixel = id.xy * 2;
    uint srcWidth , srcHeight;
    src.GetDimensions(srcWidth , srcHeight);
    const uint2 srcSize = uint2(srcWidth , srcHeight);

    const uint2 lastOffset = uint2(
        (id.x == size.x - 1 && (srcSize.x & 1) != 0) ? 2 : 1 ,
        (id.y == size.y - 1 && (srcSize.y & 1) != 0) ? 2 : 1);

    // 1 x 1 sources : there is nothing after the edge
    const uint2 srcEnd = min(srcSize - 1 , srcPixel + lastOffset);

    float result = src[srcPixel];
    for(uint y = srcPixel.y ; y <= srcEnd.y ; y++)
    {
        for(uint x = srcPixel.x ; x <= srcEnd.x ; x++)
        {
            result = max(result , src[uint2(x , y)]);
        }
    }

    dst[id.xy] = result;
}
