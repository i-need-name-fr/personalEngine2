#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(1 , 0)]] ConstantBuffer<FogVolume> fogData;
[[vk::binding(2 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(3 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardBuffer;
[[vk::binding(4 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(5 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(6 , 0)]] RWStructuredBuffer<LightSource> lightBuffer;


[[vk::image_format("rgba16f")]]
[[vk::binding(7 , 0)]] RWTexture2DArray<float4> fogImage;
[[vk::image_format("r32f")]]
[[vk::binding(8 , 0)]] RWTexture2D<float> filteredShadowImage;

float sliceDepth(const float zf){
    return FOG_NEAR * pow(FOG_FAR / FOG_NEAR , zf / fogData.gridSize );
}

// тип тут главное сделать так что Z будет Primary Образно говоря , поэтому здесь и деление такое странное 
float3 froxelWorldPos(uint3 id , uint3 fogSize){
    float2 uv = float2(float2(id.xy) + 0.5f) / float2(fogSize.xy);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    float4 v = mul(generalBuffer.inverseProj , float4(ndc , 1.0f , 1.0f));
    float3 vDis = v.xyz / v.w;
    vDis /= abs(vDis.z);
    float3 pos = vDis * sliceDepth(id.z + 0.5f );
    return mul(generalBuffer.inverseView , float4(pos , 1.0f)).xyz;
}

// в каком дальше направлении будет идти фотон через весь fog 
float phaseHG(const float cosTheta , const float g){
    float k = 1.0f + g * g - 2.0f * g * cosTheta;
    return (1.0f - g * g ) / (4.0f * PI * k * sqrt(k));
}

void findLightIndices(uint2 currentPixel , out uint firstLight , out uint lightCount){
    firstLight = 0;
    lightCount = 0;

    uint2 tileCoord = currentPixel / forwardBuffer.TILE_SIZE;
    uint tileIndex = tileCoord.x + tileCoord.y * forwardBuffer.TILE_PER_ROW;

    lightCount = min(lightCountPerTile[tileIndex] , (uint)MAX_LIGHTS);
    firstLight = tileIndex * MAX_LIGHTS;
}

[shader("compute")]
[numthreads(4 , 4 , 1)]
void main(uint3 id : SV_DispatchThreadID){
    uint width , height , tempDepth;
    fogImage.GetDimensions(width , height , tempDepth);
    uint2 screenSize = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    uint3 fogSize = uint3(width , height , tempDepth);
    if(any(id >= fogSize)) return;

    float3 pos = froxelWorldPos(id , fogSize);
    float density = fogData.density * exp(-fogData.heightFallOff * max(pos.y - fogData.fogBaseHeight , 0));   
    float sigmaT = density;
    float3 sigmaS = sigmaT * fogData.albedo;

    const float g = fogData.g;

    float3 V = normalize(pos - generalBuffer.playerPosition);
    float3 L_sun = -sunBuffer.direction;
    // the center of the froxel on the screen
    const uint2 pixel = min(uint2((float2(id.xy) + 0.5f) * float2(screenSize) / float2(fogSize.xy)) , screenSize - 1);

    uint lightOffset , lightCount;
    findLightIndices(pixel  ,lightOffset , lightCount);

    float3 Lin = sunBuffer.diffuse  * phaseHG(dot(L_sun , V) , g);
    for(uint i = 0 ; i < lightCount ; i++){
        const uint lightIndex = tileLightIndices[lightOffset + i];
        LightSource light = lightBuffer[lightIndex];
        float3 L = light.position - pos;
        float r2 = dot(L , L);
        float r = sqrt(r2);
        L  = L / r;
        float att = 1.0f / max(r2 , 0.0001f);
        float window = saturate(1.0f - pow(r / light.lightRadius , 4.0f ));
        window *= window;
        Lin += light.diffuse * window * att * phaseHG(dot(L , V) , g);
    }

    // the light of the lamps and of the sun is shadowed , the own glow of the fog is not
    const float3 scattered = Lin * filteredShadowImage[pixel].x + fogData.color;
    fogImage[id] = float4(scattered * sigmaS , sigmaT);
}
