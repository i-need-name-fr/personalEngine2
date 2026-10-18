#include "../../include/ShaderStructures/ShaderStructure.h"
[[vk::image_format("rgba32f")]]
[[vk::binding(0 , 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(1 , 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("r32f")]]
[[vk::binding(2 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("r32f")]]
[[vk::binding(3 , 0)]] RWTexture2D<float> roughnessImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(4 , 0)]] RWTexture2D<float4> viewImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(5 , 0)]] RWTexture2D<float4> diffuseImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(6 , 0)]] RWTexture2D<float4> specularImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(7 , 0)]] RWTexture2D<float4> indirectImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(8 , 0)]] RWTexture2D<float4> prevHDRImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(9 , 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("r32f")]]
[[vk::binding(10 , 0)]] RWTexture2D<float> metallicImage;

[[vk::binding(11 , 0)]] TextureCube skybox;
[[vk::binding(12 , 0)]] SamplerState typicalSampler;

[[vk::binding(13 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(14 , 0)]] ConstantBuffer<GeneralFeatures> features;

[[vk::image_format("r32f")]]
[[vk::binding(15 , 0)]] RWTexture2D<float> pixelCheck;

// Forward+ : the lamps of the tile of the pixel ( the lists are made by forward.hlsl )
[[vk::binding(16 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(17 , 0)]] RWStructuredBuffer<LightSource> lightBuffer;
[[vk::binding(18 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(19 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;

// the energy of the specular lobe that the single scattering GGX loses ( the multiple scattering LUT , the same as in LightPass )
[[vk::binding(20 , 0)]] Texture1D multiScatterTexture;

// the sun : the direction of its light and its colour
[[vk::binding(21 , 0)]] ConstantBuffer<SunSource> sunBuffer;

float calcMultiScatterFactor(float3 F0 , float roughness){
    // average of the three channels, not their sum
    const float F0_avr = (F0.x + F0.y + F0.z) / 3.0f;
    const float Ess = multiScatterTexture.SampleLevel(typicalSampler , roughness, 0).x;
    return 1.0f + F0_avr * (1.0f / max(Ess , 0.001f) - 1.0f);
}

void findLightIndices(uint2 currentPixel , out uint firstLight , out uint lightCount){
    const uint2 tileCoord = currentPixel / forwardInfo.TILE_SIZE;
    const uint tileIndex = tileCoord.x + tileCoord.y * forwardInfo.TILE_PER_ROW;

    // forward.hlsl keeps counting past MAX_LIGHTS ( it only stops WRITING there )
    lightCount = min(lightCountPerTile[tileIndex] , (uint)MAX_LIGHTS);
    firstLight = tileIndex * MAX_LIGHTS;
}


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

float3 sampleSkybox(const float3 N){
    float3 result = 0.0f;
    [loop]
    for(uint i = 0 ; i  < features.skyboxSampleCount ; i++){
        const float3 direction = randomInHemisphere(N);
        result += skybox.SampleLevel(typicalSampler , direction , 0).xyz;
    }
    result /= float(features.skyboxSampleCount);
    return result;
}


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


void computeLight(GeneralObjectInfo info , float3 L , out float3 diffuse , out float3 specular){
    const float roughness = info.roughness;
    const float metallic = info.metallic;
    const float3 V = info.V;
    const float3 N = info.N;
    const float3 albedo = info.albedo;
    const float3 H = normalize(V + L);
    const float NdotV = max(dot(N , V) , 0.0001f);
    const float VdotH = max(dot(V , H) , 0.0001f);
    const float NdotH = max(dot(N , H) , 0.0001f);
    const float NdotL = max(dot(N , L) , 0.0001f);

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f), albedo , metallic);
    const float D = D_GGX(NdotH , roughness);
    const float3 F = F_fresnel(F0 , VdotH);
    const float G = G_smith(NdotV , NdotL , roughness);

    float3 directSpecular = (D * G * F) / max(4.0f * NdotL * NdotV , 0.0001f);
    float multiScatter = calcMultiScatterFactor(F0 , roughness);
    specular = directSpecular * multiScatter;

    const float3 kD = (1.0f - F) * (1.0f - metallic);
    diffuse = kD * albedo / PI;
}

float3 stretchView(const float3 V , const float a){
    return normalize(float3(a  * V.x , V.y * a , V.z));
}

float3 sampleHemisphere(const float2 Xi){
    const float r = Xi.y;
    const float phi = 2.0f * PI * Xi.x;
    const float R = sqrt(r);
    const float t1 = cos(phi) * R;
    const float t2 = sin(phi) * R;
    const float z=  sqrt(1.0f - r);
    return float3(t1 , t2 , z);
}

float3 sampleVNDF(float2 Xi , float3 V , float roughness){
    const float alpha = roughness * roughness;
    const float3 Vh = stretchView(V , alpha);

    const float3 T1 = (Vh.z < 0.9999f) ? normalize(cross(float3(0.0f , 0.0f , 1.0f) , Vh)) : float3(1.0f , 0.0f , 0.0f);
    const float3 T2 = cross(Vh , T1);

    const float r = Xi.x;
    const float R = sqrt(r);
    const float phi = 2.0f * PI * Xi.y;
    const float t1 = cos(phi) * R;
    float t2 = sin(phi) * R;
    const float s = 0.5 * (1 + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
    const float z=  sqrt(max(0.0f , 1.0f - t1 * t1 - t2 * t2));
    const float3 Nh = t1 * T1 + t2 * T2 + z * Vh;
    return normalize(float3(alpha * Nh.x , alpha * Nh.y , max(0.0f , Nh.z)));
}

float D_GGX_exact(float NdotH , float roughness){
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float denum = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(denum * denum * PI , 1e-20f);
}

float G1_smithGGX(float NdotV , float alpha){
    const float a2 = alpha * alpha;
    return 2.0f * NdotV / (NdotV + sqrt(a2 + (1.0f - a2) * NdotV * NdotV));
}

float pdfVNDF(float3 N , float3 V , float3 L , float roughness){
    const float NdotV = dot(N , V);
    const float NdotL = dot(N , L);
    if(NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    const float3 H = normalize(V + L);
    const float NdotH = dot(N , H);
    const float alpha = roughness * roughness;
    const float safeNdotV = max(NdotV , 1e-4f);
    return G1_smithGGX(safeNdotV , alpha) * D_GGX_exact(NdotH , roughness) / (4.0f * safeNdotV);
}

float3 evalBRDF(const float3 N , const float3 V , const float3 L , const float roughness, const float3 F0){
    const float NdotV = dot(N , V);
    const float NdotL = dot(N , L);
    if(NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    const float3 H = normalize(V + L);
    const float NdotH = dot(N , H);
    const float VdotH = dot(V , H);
    const float alpha = roughness * roughness;
    // D without the cap of D_GGX ( the cap cut the highlight for roughness < 0.27 ) , G of both directions ( the shadowing of L is a part of it )
    const float D = D_GGX_exact(NdotH , roughness);
    const float G = G1_smithGGX(NdotV , alpha) * G1_smithGGX(NdotL , alpha);
    const float3 F = F_fresnel(F0 , VdotH);
    // f * cos : the NdotL of the cosine cancels the NdotL of the denominator
    return D * G * F / max(4.0f *  NdotV , 1e-20);
}

float3 evalDiffuse(const float3 albedo , const float3 N , const float3 L  ,const float metallic){
    const float3 color = albedo / PI * (1.0f - metallic);
    const float NdotL = saturate(dot(N , L));
    return color * NdotL;
}

void sampleDiffuseDirection(const float2 Xi , const float3 N , out float3 L , out float pdf){
    float3 T , B;
    buildONB(N , T , B);
    const float3 local = sampleHemisphere(Xi);
    L = normalize(local.x * T + local.y * B + local.z * N);
    pdf = max(dot(N , L) , 0.0f) / PI;
}

void sampleSpecularDirection(const float2 Xi , const float3 N , const float3 V , const float roughness , out float3 L , out float pdf){
    float3 T , B;
    buildONB(N , T , B);
    const float3 Vlocal = float3(dot(V , T) , dot(V , B) , dot(V , N));

    L = N;
    pdf = 0.0f;
    // a view from behind the surface ( an interpolated normal ) : no valid sample
    if(Vlocal.z <= 0.0f) return;

    const float3 Hlocal = sampleVNDF(Xi , Vlocal , roughness);
    const float3 Llocal = normalize(reflect(-Vlocal , Hlocal));
    L = normalize(Llocal.x * T + Llocal.y * B + Llocal.z * N);
    // pdfVNDF works in any space as long as N , V and L are in the same one : the tangent one here ( it is 0 for L under the surface )
    pdf = pdfVNDF(float3(0.0f , 0.0f , 1.0f) , Vlocal , Llocal , roughness);
}


float3 computeSSGI(uint2 currentPixel , out float visibility){

    // random : 8 rays per pixel , the directions come from a seed of the pixel and the frame
    // ( the no random version used the points of a Hammersley set , see the commented line in the loop )
    const uint SAMPLES = 2u;
    const float3 P = positionImage[currentPixel].xyz;
    const float3 N = normalImage[currentPixel].xyz;
    const float3 cameraPos = generalBuffer.playerPosition;
    const float3 V = normalize(generalBuffer.playerPosition - P);
    const float roughness = roughnessImage[currentPixel].x;
    const float metallic = metallicImage[currentPixel].x;
    uint width , height;
    prevHDRImage.GetDimensions(width, height);
    const uint2 screenSize = uint2(width,  height);
    uint seed = 8345u * currentPixel.x * 24u * currentPixel.y + generalBuffer.frameIndex * 2398u;
    // so now instead of simply taking the prev HDR image result , we will be doing smth new
    float hitDistance = 0.0f;
    float3 result = 0.0f;
    

    {
        // first we send the direction
        const float3 H = GGX_halfVector(seed , N , roughness);
        const float3 dir = normalize(reflect(-V , H));
        const float3 offset = P + N * 1e-6f;
        Hit data = SSTraceRay(offset , dir , generalBuffer.playerPosition ,0.1f , 32 );
        if(data.hit){


            // here if we hit now the magic starts
            // first we need to find the position where did we hit
            const uint2 id = uint2(data.uv * float2(screenSize) - 0.5f);
            const float3 samplePos = positionImage[id].xyz;
            hitDistance = length(P - samplePos);
            // now we find the light indices and count 

            uint lightCount , lightOffset;
            findLightIndices(id , lightOffset , lightCount);

            GeneralObjectInfo info = (GeneralObjectInfo)0;
            info.P = samplePos;
            info.metallic = metallicImage[id.xy];
            info.N = normalImage[id.xy].xyz;
            info.V = normalize(generalBuffer.playerPosition - samplePos);
            info.roughness = roughnessImage[id];
            info.albedo = albedoImage[id].xyz;
            
            if(lightCount > 0){

                const uint lightID = (uint)(rand(seed) * lightCount) + lightOffset;
                if(lightID != 9999){
                    LightSource light = lightBuffer[tileLightIndices[lightID]];
                    
                    float3 L = light.position - samplePos;
                    const float d = length(L);
                    L = normalize(L);

                    const float att = 1.0f / ( 1.0f + light.lightLinear * d + d * d * light.lightQuadratic);
                    float3 diffuse , specular;
                    computeLight(info , L , diffuse , specular);
                    diffuse /= max(info.albedo , 1e-20f);

                    result += (diffuse) * light.diffuse * att * saturate(dot(L , normalImage[id].xyz))  * lightCount;
                }
            }

            // now we calculate the sun effection to the
            {
                float3 L = -sunBuffer.direction;
                float3 diffuse , specular;
                computeLight(info , L , diffuse , specular);
                diffuse /= max(info.albedo , 1e-20f);

                const float3 fcos = evalDiffuse(info.albedo , info.N , L , info.metallic);
                result += (diffuse) * sunBuffer.diffuse * saturate(dot(L , N)) * fcos;
            }

        }else{
            const float3 F0 = lerp(float3(0.04 , 0.04 , 0.04) , albedoImage[currentPixel].xyz , metallic);
            const float3 V = normalize(generalBuffer.playerPosition - positionImage[currentPixel].xyz);
            const float NdotV = max(dot(normalImage[currentPixel].xyz , V) , 0.0001f);
            const float3 F = F_fresnel(F0 , NdotV);
            const float3 kD = (1.0f - F) * (1.0f - metallic);
            result =  sampleSkybox(normalImage[currentPixel].xyz) * kD;
            hitDistance = 999ul;
        }
    }
    visibility = 0.0f;
    diffuseImage[currentPixel].a = hitDistance;
    specularImage[currentPixel].a = hitDistance;
    indirectImage[currentPixel].a = hitDistance;

    float3 reflectedDir = reflect(-V , N);
    viewImage[currentPixel] = float4( P + reflectedDir * hitDistance , 1.0f);
    return result;
}


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
    if(pixelCheck[id.xy] != COLOR_PIXEL){
        // we dont have to set hit distance , since the denoisers filter the skybox and non skybox pixels 
        // so the only requirement is to return , since it wont filter the indirect image too 
        // NOTE : however maybe we need to set at least float3(0.0f) , since denoisers need history
        indirectImage[id.xy] = 0.0f;
        return;
    }
    // the sky : the G-buffer is cleared to 0 there , and diffuseImage holds the colour of the sky itself ( PostProcess copies it to the
    // picture ) . the indirect light is added into diffuseImage now , and for the albedo 0 of the sky it is divided by 0.02 :
    // the whole sky went white
    if(dot(normalImage[id.xy].xyz , normalImage[id.xy].xyz) < 0.5f) return;

    uint seed = initSeed(id.xy , generalBuffer.frameIndex);
    // the numbers of the directions of SSGI ( randNoise ) : the pixel and the frame
    initNoise(id.xy , generalBuffer.frameIndex);

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

        // the reflection is specular light : the multiple scattering factor of the lobe ( F0 and roughness of the pixel ) , like in LightPass
        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , albedoImage[id.xy].xyz , metallicImage[id.xy]);
        reflectedColor *= calcMultiScatterFactor(F0 , roughness);

    }

    const float3 result = lerp(reflectedColor , indirectLight , roughness);

    // the ambient light of the sky ( the irradiance cube , the same as in LightPass ) times what the pixel sees of the sky . in the
    // demodulated form : LightPass ambient = irradiance * kD * albedo , and the albedo comes back in PostProcess
    float3 skyLight = float3(0.0f , 0.0f , 0.0f);

    // no weight of the indirect light for now ( indirectWeight is not applied : the clean colour )
    // the debug picture of the indirect light shows all of it : the bounces , the reflections and the sky light
    indirectImage[id.xy] = float4(result + skyLight * albedoImage[id.xy].xyz , 1.0f);
}