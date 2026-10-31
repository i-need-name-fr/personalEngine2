[[vk::image_format("rgba16f")]]
[[vk::binding(0 , 0 )]] RWTexture2D<float4> hdrColorImage;
[[vk::binding(1 , 0)]] RWStructuredBuffer<uint> luminanceHistory;

float luminance(float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }

static float minLogLum = -10.0f;
static float logLumRange = 12.0f;

uint ColorToHistoryLuminance(float3 color){
    float lum = luminance(color);
    // written as !( lum >= x ) , not ( lum < x ) : a NaN pixel fails every comparison , and a NaN
    // would go through clamp and uint() to an arbitrary index of a 256 uint buffer ( out of bounds write )
    if(!(lum >= 0.0005f)) return 0;

    float logLum = clamp((log2(lum) - minLogLum) / logLumRange, 0.0f , 1.0f  );
    return min(uint(logLum * 254 + 1.0f) , 255u);   
}


[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    hdrColorImage.GetDimensions(width , height);
    if(id.x >= width || id.y >= height) return;

    const float3 currentColor = hdrColorImage[id.xy].rgb;
    uint index = ColorToHistoryLuminance(currentColor);
    InterlockedAdd(luminanceHistory[index] , 1);
}