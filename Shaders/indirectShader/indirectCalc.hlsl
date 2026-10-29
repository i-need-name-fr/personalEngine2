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
[[vk::binding(17 , 0)]] RWStructuredBuffer<lg::LightSource> lightBuffer;
[[vk::binding(18 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(19 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;

// the energy of the specular lobe that the single scattering GGX loses ( the multiple scattering LUT , the same as in LightPass )
[[vk::binding(20 , 0)]] Texture1D multiScatterTexture;

// the sun : the direction of its light and its colour
[[vk::binding(21 , 0)]] ConstantBuffer<SunSource> sunBuffer;

// set 1 : the world grid of the lamps ( the descriptor buffer of the LightManager ) . its binding 0 is the lamps again : lightBuffer above is the same buffer
[[vk::binding(1 , 1)]] RWStructuredBuffer<lg::LightCell> lightCells;
[[vk::binding(2 , 1)]] ConstantBuffer<lg::GridCascadeInfo> gridInfo;

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

// ---------------------------------------------------------------- the world grid of the lamps ( the same functions as in generalHit.hlsl )
inline float3 getGridOrigin(const float3 playerPosition) {
    const float3 snapped = floor(playerPosition / lg::GRID_CELL_SIZE) * lg::GRID_CELL_SIZE;
    return snapped - float3(lg::GRID_SIZE * 0.5f * lg::GRID_CELL_SIZE, lg::GRID_SIZE * 0.5f * lg::GRID_CELL_SIZE,lg::GRID_SIZE * 0.5f * lg::GRID_CELL_SIZE);
}

inline float3 translateToGridPos(const float3 position , const float3 playerPosition) {
    return position - getGridOrigin(playerPosition);
}

inline int32_t translatePosToCellIndex(const float3 gridPos) {
    const float3 cell = floor(gridPos / lg::GRID_CELL_SIZE);
    const float n = lg::GRID_SIZE;
    if(cell.x < 0.0f || cell.y < 0.0f || cell.z < 0.0f || cell.x >= n || cell.y >= n || cell.z >= n) return NON_VALID_SLOT;

    const int32_t count = int(lg::GRID_SIZE);
    return int32_t(cell.x) + count * (int32_t(cell.y) + count * int32_t(cell.z));
}

int findGridSlot(const float3 P){
    const float3 gridPos = translateToGridPos(P , generalBuffer.playerPosition);
    return translatePosToCellIndex(gridPos);
}

// the lamps of the cell of the world position P ( not of a pixel tile , so it works for a point that is not on the screen )
void findLightIndices(const float3 P , out uint lightIndices[lg::MAX_LIGHT_PER_CELL] ,out uint lightCount){
    lightCount = 0;
    const int gridID = findGridSlot(P);
    if(gridID == NON_VALID_SLOT) return;

    const lg::LightCell cell = lightCells[gridID];
    // the grid shader keeps counting past MAX_LIGHT_PER_CELL ( it only stops WRITING there )
    lightCount = min(cell.totalLightCount , (uint)lg::MAX_LIGHT_PER_CELL);
    lightIndices = cell.lightIndices;
}


// the farther the thing that gives the light is ( the hit distance ) , the less it matters : 1 at the surface , 0.5 at
// INDIRECT_FALLOFF_DISTANCE , and then it falls like 1 / d^2 . SSDO and SSGI multiply what a sample brings by it
static const float INDIRECT_FALLOFF_DISTANCE = 8.0f;
static const float SHADOW_STEP_SIZE = 0.5f;

// SSGI ray : the first step is 0.2 m and every next step is 8% longer than the last ( 64 steps reach ~ 100 m ) , the ray stops at SSGI_MAX_DISTANCE
static const float SSGI_STEP_SIZE = 0.2f;
static const float SSGI_STEP_GROWTH = 1.08f;
static const float SSGI_MAX_DISTANCE = 40.0f;
// a ray that missed and did NOT reach the sky on the screen ( it left the screen or ran out of distance ) : the sky behind it is unknown ,
// indoors it is usually closed . the sky colour is multiplied by this ( 1 = believe the sky , 0 = black ) , tune by eye
static const float SSGI_UNKNOWN_SKY_WEIGHT = 0.25f;

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
    // the ray reached a pixel of the sky : nothing is in front of it on the screen , so the sky really is there .
    // ( a miss without it : the ray left the screen or ran out of distance , and what is behind is unknown )
    bool sky;
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


// stepGrowth > 1 : every step is longer than the last ( near the surface the ray is checked finely , far away coarsely ) , so a long ray costs ~40 steps and not 500
Hit SSTraceRay(float3 P , float3 N , float3 dir , float3 cameraPos , float stepSize , const float dist , const float stepGrowth = 1.0f){

    Hit info;
    info.hit = false;
    info.sky = false;
    info.uv = float2(0.0f , 0.0f);
    info.distance = 0.0f;

    uint width , height;
    prevHDRImage.GetDimensions(width, height);
    uint2 screenSize = uint2(width,  height);

    // how far behind the visible surface the ray may be and still count as a hit : the screen only knows the front of things ,
    // a ray that is far behind a surface is really somewhere else ( it passed behind a pole , through a gap )
    const float thickness = 2.0f;

    const uint maxSteps = (stepGrowth > 1.0f) ? 64u : uint(floor(dist / stepSize));

    float t = 0.0f;
    float step = stepSize;

    [loop]
    for(uint i = 1 ; i < maxSteps ; i++){   // step 0 is the pixel itself

        t += step;
        step *= stepGrowth;
        if(t > dist) break;
        float3 currentP = P + dir * t;
        float4 clip = mul(generalBuffer.viewProj , float4(currentP , 1.0f));
        // behind the camera : no pixel there
        if(clip.w <= 1e-4f) break;
        float2 uv = (clip.xy / clip.w) * float2(0.5f , -0.5f) + 0.5f;

        if(any(uv >= 1.0f ) || any(uv < 0.0f)) break;
        // the old uv * size - 0.5 wrapped to a huge uint on the first pixel row / column
        uint2 pPixel = min(uint2(uv * float2(screenSize)) , screenSize - 1u);

        // the sky : its position is 0 , it looked like a surface at the other side of the world and every ray that reached it was a "hit"
        // with the colour of the sky
        const float3 pN = normalImage[pPixel].xyz;
        if(dot(pN , pN) < 0.5f){
            // the ray is over the sky : only believe it for rays that go up , a ray that goes down and sees "sky" is just passing a gap
            if(dir.y > 0.0f){ info.sky = true; break; }
            continue;
        }
        // the surface that looks away from the ray : the ray went behind its front side ( the screen knows only the front ) , it is not a real hit .
        // ( the normals are NOT compared with N : a wall and a floor are perpendicular , and that is where the bounce matters )
        if(dot(pN , dir) > 0.0f) continue;

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

bool computeShadow(const float3 P , const float3 N , const float3 dir , const float distance)
{
    Hit shadow = SSTraceRay(offsetRay(P , N , generalBuffer.playerPosition),  N , dir , generalBuffer.playerPosition, SHADOW_STEP_SIZE , distance);
    return shadow.hit;
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

bool isSpecularMode(const float roughness , const float metallic){ return (metallic >= 0.7f || roughness < 0.4f); }

GeneralObjectInfo readGBuffer(const uint2 pixel, const float3 P)
{
    GeneralObjectInfo info;

    info.N = normalImage[pixel].xyz;
    info.P = positionImage[pixel].xyz;
    info.roughness = roughnessImage[pixel].x;
    info.metallic = metallicImage[pixel].x;
    info.albedo = albedoImage[pixel].xyz;
    info.V = normalize(P - info.P);

    return info;
}

float3 computeLightAt(GeneralObjectInfo data, inout uint seed){
    // in order to literally minimaze the atmosphere effect , we actually need to lower the chances of it to appear * 
    uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
    uint2 pixel = DispatchRaysIndex().xy;

    findLightIndices(data.P , lightIndices , lightCount);

    
    const float3 saveAlbedo = max(data.albedo , 1e-20);

    float3 result = 0.0f ;
    if(lightCount){
        const uint id = (uint)(rand(seed) * lightCount);
        lg::LightSource light = lightBuffer[lightIndices[id]];
        float3 diffuse, specular;

        const float3 V = data.V;
        data.V = V;
        float3 L = light.position - data.P;
        L = normalize(L);
        // the centre of the lamp , not a random point of its sphere ( the indirect light : a big sphere reached through the walls )
        float3 lightPos = light.position;
        float3 realL = lightPos - data.P;
        const float dist = length(realL);
        realL = normalize(realL);
        
        computeLight(data ,realL , diffuse , specular );

        const float NdotL = saturate(dot(data.N , realL));  
        float shadow = (computeShadow(data.P , data.N , L , dist) ? 0.0f : 1.0f);
        diffuse /= saveAlbedo;
        result = NdotL * light.diffuse * (diffuse) * shadow  * lightCount;
    }else{
        result =  0.01f;
    }
    if(sunBuffer.intensity > 0.0f){
        float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
        // now we also have to check if the shadow actually hits it , since if not, there is no point at computing the light , since its also the calcualtion
        const float3 L = -sunBuffer.direction;
        // under the open sky the shadow of the sun is traced , in the other case the sun gives nothing ( 0 )
        float sunShadow = (computeShadow(data.P , data.N , L , 100.0f) ? 0.0f : 1.0f);
        computeLight(data , L , sunDiffuse , sunSpecular);
        sunDiffuse /= saveAlbedo;
        result += (sunDiffuse) *  clamp(dot(data.N , L),  0.0f , 1.0f) * sunBuffer.diffuse  * sunShadow;
    }

    result *= saveAlbedo;

    return result;
}

// the light that leaves the hit of a screen space ray : it is read from the picture of the LAST frame ( prevHDRImage ) , not computed again . the picture has the direct light ,
// the bounces and the reflections of the hit already . the hit is moved to the last frame with prevViewProj ( it could have moved on the screen ) ;
// the place of this frame is used when the hit was outside of the last picture
float3 readHitRadiance(const uint2 hitPixel , const float2 hitUV)
{
    const uint2 size = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);

    float2 prevUV = hitUV;
    const float4 clip = mul(generalBuffer.prevViewProj , float4(positionImage[hitPixel].xyz , 1.0f));
    if(clip.w > 1e-4f){
        const float2 reprojected = (clip.xy / clip.w) * float2(0.5f , -0.5f) + 0.5f;
        if(all(reprojected >= 0.0f) && all(reprojected < 1.0f)) prevUV = reprojected;
    }

    const uint2 prevPixel = min(uint2(prevUV * float2(size)) , size - 1u);
    float3 radiance = prevHDRImage[prevPixel].xyz;
    // one NaN / Inf in the picture goes through the denoiser into the history
    if(any(isnan(radiance)) || any(isinf(radiance))) radiance = 0.0f;
    return radiance;
}

void computeGI(const uint2 pixel , GeneralObjectInfo info, inout uint seed , out float3 result , out float distance)
{

    distance = 0.0f;
    result = 0.0f;

    const uint2 size = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    const bool needsSpec = isSpecularMode(info.roughness, info.metallic);
    float3 L =0.0f; float pdf = 0.0f;

    if(!needsSpec)
    {
        // float3 P , float3 N , float3 dir , float3 cameraPos , float stepSize , const float dist 
        sampleDiffuseDirection(float2(randNoise() , randNoise()), info.N , L , pdf);
        if(pdf > 0.0f)
        {
            Hit data = SSTraceRay(offsetRay(info.P , info.N , generalBuffer.playerPosition) , info.N , L , generalBuffer.playerPosition , SSGI_STEP_SIZE , SSGI_MAX_DISTANCE , SSGI_STEP_GROWTH);
            if(data.hit)
            {

                const uint2 pPixel = min(uint2(data.uv * float2(size)) , size - 1u);
                const float3 fcos = evalDiffuse(info.albedo, info.N , L , info.metallic);
                const GeneralObjectInfo hitData = readGBuffer(pPixel , info.P);
                result = computeLightAt(hitData , seed) * fcos / pdf;
                result /= max(info.albedo  , 1e-20);
                distance = data.distance;
            }
            else
            {

                const float3 skyV = L;
                const float3 skyF0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
                const float skyNdotV = max(dot(info.N , skyV) , 0.0001f);
                const float3 skyF = F_fresnel(skyF0 , skyNdotV);
                const float3 skyKD = (1.0f - skyF) * (1.0f - info.metallic);
                distance = 990.0f;
                result = skybox.SampleLevel(typicalSampler , L , 0).xyz * skyKD * (data.sky ? 1.0f : SSGI_UNKNOWN_SKY_WEIGHT);
            }
        }

    }
    else
    {

        sampleSpecularDirection(float2(randNoise() , randNoise()), info.N , info.V , info.roughness , L , pdf);
        if(pdf > 0.0f)
        {
            Hit data = SSTraceRay(offsetRay(info.P , info.N , generalBuffer.playerPosition) , info.N , L , generalBuffer.playerPosition , SSGI_STEP_SIZE , SSGI_MAX_DISTANCE , SSGI_STEP_GROWTH);
            if(data.hit)
            {

                const uint2 pPixel = min(uint2(data.uv * float2(size)) , size - 1u);
                const float3 F0 = lerp(0.04f , info.albedo , info.metallic);
                const float3 fcos = evalBRDF(info.N , info.V , L , info.roughness , F0);
                const GeneralObjectInfo hitData = readGBuffer(pPixel , info.P);
                result = computeLightAt(hitData , seed) * fcos / pdf;
                distance = data.distance;
            }
            else
            {

                const float3 skyV = L;
                const float3 skyF0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
                const float skyNdotV = max(dot(info.N , skyV) , 0.0001f);
                const float3 skyF = F_fresnel(skyF0 , skyNdotV);
                const float3 skyKD = (1.0f - skyF) * (1.0f - info.metallic);
                distance = 990.0f;
                result = skybox.SampleLevel(typicalSampler , L , 0).xyz * skyKD * (data.sky ? 1.0f : SSGI_UNKNOWN_SKY_WEIGHT);
            }
        }

    }
}



float3 SSTraceRayReflection(const GeneralObjectInfo info , const float3 direction , inout uint seed , out float hitDistance){
    const uint2 size = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    const float3 origin = offsetRay(info.P , info.N , generalBuffer.playerPosition);

    const Hit data = SSTraceRay(origin , info.N , direction , generalBuffer.playerPosition , SSGI_STEP_SIZE , SSGI_MAX_DISTANCE , SSGI_STEP_GROWTH);
    if(!data.hit){
        // a miss is the sky ( generalMiss.hlsl ) : a mirror gives it back as it is , the Fresnel is in fcos of the caller
        hitDistance = 990.0f;
        return skybox.SampleLevel(typicalSampler , direction , 0).xyz;
    }

    // the old uv * size - 0.5 wrapped to a huge uint on the first pixel row / column
    const uint2 hitPixel = min(uint2(data.uv * float2(size)) , size - 1u);
    // V of the hit is the direction back to the surface that the ray comes from , like -WorldRayDirection() in Reflection::readHitData
    const GeneralObjectInfo hitData = readGBuffer(hitPixel , info.P);

    hitDistance = data.distance;
    return computeLightAt(hitData , seed);
}

// the screen space reflection of a pixel : the same block as "if(needSpecular && !load.fromReflection)" of processPixel in generalHit.hlsl
void computeSSR(const GeneralObjectInfo info , inout uint seed , out float3 result , out float distance){
    result = 0.0f;
    distance = 0.0f;

    float3 L = 0.0f;
    float pdf = 0.0f;
    sampleSpecularDirection(float2(randNoise() , randNoise()) , info.N , info.V , info.roughness , L , pdf);
    if(pdf <= 0.0f) return;

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
    const float3 fcos = evalBRDF(info.N , info.V , L , info.roughness , F0);

    const float3 radiance = SSTraceRayReflection(info , L , seed , distance);
    float3 reflectedColor = radiance * fcos / pdf;
    // the lobe of a metal is narrow : fcos and pdf are both huge there , and one NaN / Inf goes through the denoiser into the history ( the whole specular goes black )
    if(any(isnan(reflectedColor)) || any(isinf(reflectedColor))) reflectedColor = 0.0f;

    result = reflectedColor;
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
        // the sky has no position in the G buffer : the direction of its pixel comes from the camera matrix ( the point at the middle depth lies on the ray of the pixel ,
        // in front of the camera, for either direction of the depth ) , and V looks BACK to the camera like everywhere else
        const float2 skyUV = (float2(id.xy) + 0.5f) / float2(screenSize);
        const float3 skyDir = normalize(reconstructWorldPos(skyUV , 0.5f , generalBuffer.invViewProj) - generalBuffer.playerPosition);
        viewImage[id.xy] = float4(-skyDir , 1.0f);
        return;
    }
    float3 reflectedColor = 0.0f;
    float3 indirectLight = 0.0f;
    float sd = 0.0f;
    float dd = 0.0f;

    initNoise(id.xy , generalBuffer.frameIndex);
    GeneralObjectInfo info = readGBuffer(id.xy , generalBuffer.playerPosition);
    const bool spec = isSpecularMode(info.roughness ,info.metallic);

    // the direction from the surface to the camera : ReLAX_temporal moves the point by it ( worldPos - V * hitDistance ) to find the virtual position of a reflection
    viewImage[id.xy] = float4(info.V , 1.0f);

    uint seed = id.x * 3984u + id.y * 8324u + generalBuffer.frameIndex * 348u;

    if(spec)
    {
        computeSSR(info , seed , reflectedColor ,sd);
    }

    computeGI(id.xy , info , seed , indirectLight , dd);
    specularImage[id.xy].xyz += reflectedColor;
    specularImage[id.xy].a = sd;
    diffuseImage[id.xy].a = dd;
    indirectImage[id.xy] = float4(indirectLight , dd);
}