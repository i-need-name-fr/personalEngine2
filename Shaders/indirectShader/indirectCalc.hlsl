#include "../../include/ShaderStructures/ShaderStructure.h"
[[vk::image_format("rgba32f")]]
[[vk::binding(1 , 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(2 , 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("r32f")]]
[[vk::binding(3 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("r32f")]]
[[vk::binding(4 , 0)]] RWTexture2D<float> roughnessImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(5 , 0)]] RWTexture2D<float4> viewImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(6 , 0)]] RWTexture2D<float4> diffuseImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(7 , 0)]] RWTexture2D<float4> specularImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(8 , 0)]] RWTexture2D<float4> indirectImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(9 , 0)]] RWTexture2D<float4> prevHDRImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(10 , 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("r32f")]]
[[vk::binding(11 , 0)]] RWTexture2D<float> metallicImage;

[[vk::binding(12 , 0)]] TextureCube skybox;
[[vk::binding(13 , 0)]] SamplerState typicalSampler;

[[vk::binding(14 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(15 , 0)]] ConstantBuffer<GeneralFeatures> features;


// the farther the thing that gives the light is ( the hit distance ) , the less it matters : 1 at the surface , 0.5 at
// INDIRECT_FALLOFF_DISTANCE , and then it falls like 1 / d^2 . SSDO and SSGI multiply what a sample brings by it
static const float INDIRECT_FALLOFF_DISTANCE = 8.0f;

float indirectDistanceWeight(float hitDistance){
    const float ratio = hitDistance / INDIRECT_FALLOFF_DISTANCE;
    return 1.0f / (1.0f + ratio * ratio);
}



float3 reconstructWorldPos(float2 uv , float depth , float4x4 invViewProj){
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    float4 pos = mul(invViewProj , float4(ndc , depth , 1.0f));
    return pos.xyz / pos.w;
}

struct Hit{
    float2 uv;
    float distance;
    bool hit;
};

Hit SSTraceRay(float3 P , float3 dir , float3 cameraPos , float stepSize , uint maxSteps ){

    Hit info;
    info.hit = false;
    info.uv = float2(0.0f , 0.0f);
    info.distance = 0.0f;

    uint width , height;

    prevHDRImage.GetDimensions(width, height);
    uint2 screenSize = uint2(width,  height);

    // how far behind the visible surface the ray may be and still count as a hit : the screen only knows the front of things ,
    // a ray that is far behind a surface is really somewhere else ( it passed behind a pole , through a gap )
    const float thickness = 2.0f;

    [unroll]
    for(uint i = 1 ; i < maxSteps ; i++){   // step 0 is the pixel itself

        float3 currentP = P + dir * stepSize * (float)i;
        float4 clip = mul(generalBuffer.viewProj , float4(currentP , 1.0f));
        // behind the camera : no pixel there
        if(clip.w <= 1e-4f) break;
        float2 uv = (clip.xy / clip.w) * float2(0.5f , -0.5f) + 0.5f;

        if(any(uv >= 1.0f ) || any(uv < 0.0f)) break;
        // the old uv * size - 0.5 wrapped to a huge uint on the first pixel row / column
        uint2 pPixel = min(uint2(uv * float2(screenSize)) , screenSize - 1u);

        // the sky : its position is 0 , it looked like a surface at the other side of the world and every ray that reached it was a "hit"
        // with the colour of the sky
        if(dot(normalImage[pPixel].xyz , normalImage[pPixel].xyz) < 0.5f) continue;

        float3 pP = positionImage[pPixel].xyz;

        float rayDist = length(currentP - generalBuffer.playerPosition);
        float sceneDist = length(pP - generalBuffer.playerPosition);

        float diff = rayDist - sceneDist;
        if(diff > 0.02f && diff < thickness){
            info.hit = true;
            info.distance = length(pP - P);
            info.uv = uv;
            // the FIRST surface the ray goes behind : it used to keep marching and take the last step that was behind something
            break;
        }
    }

    if(!info.hit) info.distance = 2000.0f;

    return info;
}

float3 computeSSGI(uint2 currentPixel , out float visibility){

    // random : 8 rays per pixel , the directions come from a seed of the pixel and the frame
    // ( the no random version used the points of a Hammersley set , see the commented line in the loop )
    const uint SAMPLES = 8u;

    float3 result = float3(0.0f , 0.0f , 0.0f);
    float distanceSum = 0.0f;
    float occlusion = 0.0f;

    const float3 P = positionImage[currentPixel].xyz;
    const float3 N = normalImage[currentPixel].xyz;
    const float3 cameraPos = generalBuffer.playerPosition;
    const float3 V = normalize(generalBuffer.playerPosition - P);
    const float roughness = roughnessImage[currentPixel].x;

    uint width , height;
    prevHDRImage.GetDimensions(width, height);
    const uint2 screenSize = uint2(width,  height);

    uint seed = initSeed(currentPixel , generalBuffer.frameIndex);

    for(uint i = 0u ; i < SAMPLES ; i++){
        // the no random version :
        // const float3 H = GGX_halfVectorFromXi(Hammersley(i , SAMPLES) , N , roughness);
        const float3 H = GGX_halfVector(seed , N , roughness);
        const float3 dir = normalize(reflect(-V , H));
        Hit info = SSTraceRay(P , dir , cameraPos , 0.5f , 16);

        if(info.hit){
            uint2 pPixel = uint2(info.uv * float2(screenSize) - 0.5f);
            // the farther the hit is , the less its light counts
            // capped like in SSDO : a hot pixel ( a light , a highlight ) must not light the whole neighbourhood
            result += min(prevHDRImage[pPixel].xyz , float3(8.0f , 8.0f , 8.0f)) * indirectDistanceWeight(info.distance);
            // a ray that hit something does not reach the sky
            occlusion += indirectDistanceWeight(info.distance);
        }

        // a miss can report a huge distance , it must not blow the average up
        distanceSum += min(info.distance , 10000.0f);
    }

    result /= (float)SAMPLES;
    visibility = saturate(1.0f - occlusion / (float)SAMPLES);
    const float hitDistance = distanceSum / (float)SAMPLES;

    diffuseImage[currentPixel].a = hitDistance;
    specularImage[currentPixel].a = hitDistance;
    indirectImage[currentPixel].a = hitDistance;

    float3 reflectedDir = reflect(-V , N);
    viewImage[currentPixel] = float4( P + reflectedDir * hitDistance , 1.0f);
    result *= albedoImage[currentPixel].xyz;
    return result;
}




// SSR
// SSR is the last one 
// NOTE:
// SSR will be the one writing data in viewImage and hit distance
// since it is more acquired ngl
// NOTE 2:
// if SSR is disabled , other will come in clatch
// you cannot disable all , one will be remained


float3 computeSSR(uint seed , uint2 currentPixel){

    float3 result = float3(0.0f , 0.0f , 0.0f);
    const float3 P = positionImage[currentPixel].xyz;
    const float3 N = normalImage[currentPixel].xyz;
    const float3 cameraPos = generalBuffer.playerPosition;
    const float3 V = normalize(cameraPos - P);
    // i dont want it to have to much noise
    const float3 dir = normalize(reflect(-V , N));
    Hit info = SSTraceRay(P , dir , cameraPos , 0.5f , 16);

    if(info.hit){
        uint width , height;

        prevHDRImage.GetDimensions(width, height);
        uint2 screenSize = uint2(width,  height);
        uint2 pPixel = uint2(info.uv * float2(screenSize) - 0.5f);
        result = prevHDRImage[pPixel].xyz;

    }else{
        result = skybox.SampleLevel(typicalSampler , dir , 0).xyz;
    }

    // what is reflected is the picture of the last frame , which already holds the reflections of the frame before : without a cap a bright
    // spot feeds itself ( a mirror gives back all of it every frame )
    result = min(result , float3(16.0f , 16.0f , 16.0f));

    diffuseImage[currentPixel].a = info.distance;
    specularImage[currentPixel].a = info.distance;
    indirectImage[currentPixel].a = info.distance;

    viewImage[currentPixel] = float4( P + dir * info.distance, 1.0f);

    return result;
}

[shader("compute")]
[numthreads(8,  8, 1)]
void main(uint3 id : SV_DispatchThreadID){
    
    uint width , height;
    prevHDRImage.GetDimensions(width, height);
    uint2 screenSize = uint2(width,  height);
    uint2 currentPixel = id.xy;
    if(any(currentPixel >= screenSize)) return;

    // the sky : the G-buffer is cleared to 0 there , and diffuseImage holds the colour of the sky itself ( PostProcess copies it to the
    // picture ) . the indirect light is added into diffuseImage now , and for the albedo 0 of the sky it is divided by 0.02 :
    // the whole sky went white
    if(dot(normalImage[id.xy].xyz , normalImage[id.xy].xyz) < 0.5f) return;

    uint seed = initSeed(id.xy , generalBuffer.frameIndex);

    float3 reflectedColor = float3(0.0f , 0.0f,  0.0f);
    float3 indirectLight = reflectedColor; 

    const float roughness = roughnessImage[id.xy];
    bool totalReflection = roughness ==0.0f;

    // the part of the sky that the pixel sees : the rays that escape bring the ambient light , the ones that hit something bring that
    // something's light ( and not the sky ) . a mirror pixel does not cast rays , it sees all of it
    float skyVisibility = 1.0f;

    if(features.indirectMode == SSGI_REQUIRED && !totalReflection){

        indirectLight = computeSSGI(id.xy , skyVisibility);
    }

    if(features.indirectMode == SSR_REQUIRED){

        reflectedColor = computeSSR(seed , id.xy);

    }

    const float3 result = lerp(reflectedColor , indirectLight , roughness);

    // the ambient light of the sky ( the irradiance cube , the same as in LightPass ) times what the pixel sees of the sky . in the
    // demodulated form : LightPass ambient = irradiance * kD * albedo , and the albedo comes back in PostProcess
    float3 skyLight = float3(0.0f , 0.0f , 0.0f);
    if(features.indirectMode == SSGI_REQUIRED || features.indirectMode == SSR_REQUIRED){
        const float3 N = normalImage[id.xy].xyz;
        const float3 albedo = albedoImage[id.xy].xyz;
        const float metallic = metallicImage[id.xy];
        const float3 V = normalize(generalBuffer.playerPosition - positionImage[id.xy].xyz);
        const float NdotV = max(dot(N , V) , 0.0001f);
        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , albedo , metallic);
        const float3 F = F_fresnel(F0 , NdotV);
        const float3 kD = (1.0f - F) * (1.0f - metallic);
        skyLight = skybox.SampleLevel(typicalSampler , N , 0).xyz * kD * skyVisibility;
    }

    // the balance of the picture : the indirect light has indirectWeight ( the direct one has directWeight in LightPass / PostProcess )
    // the debug picture of the indirect light shows all of it : the bounces , the reflections and the sky light
    indirectImage[id.xy] = float4((result + skyLight * albedoImage[id.xy].xyz) * generalBuffer.indirectWeight , 1.0f);

    // the indirect light is noisy ( random samples per pixel ) , and PostProcess used to add it to the picture as it was : the denoisers only
    // saw the direct light . added into diffuse here it is denoised with it ( .a of diffuse is the hit distance , it stays )
    float4 diffuseWithIndirect = diffuseImage[id.xy];
    // divided by the albedo like the direct diffuse ( PostProcess multiplies it back ) , so the sum is unchanged
    diffuseWithIndirect.rgb += (result / max(albedoImage[id.xy].xyz , 0.02f) + skyLight) * generalBuffer.indirectWeight;
    diffuseImage[id.xy] = diffuseWithIndirect;
}