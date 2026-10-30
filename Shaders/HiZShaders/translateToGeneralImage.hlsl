#include "../../include/ShaderStructures/ShaderStructure.h"

// push descriptor set : 0 = the z buffer ( D32 , sampled ) , 1 = the mip 0 of the hi Z ( R32 , storage )
[[vk::binding(0 , 0)]] Texture2D<float> depthBuffer;

[[vk::image_format("r32f")]]
[[vk::binding(1 , 0)]] RWTexture2D<float> fullImage;

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width , height;
    fullImage.GetDimensions(width , height);
    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    // now we simply transfer the data
    fullImage[id.xy] = depthBuffer.Load(int3(id.xy , 0));
}
