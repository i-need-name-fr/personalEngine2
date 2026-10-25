#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RaytracingAccelerationStructure TLAS;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(3 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(4 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(5 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(6 , 0)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(7 , 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(8 , 0)]] RWStructuredBuffer<InstanceData> instanceLayout;
[[vk::binding(9 , 0)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(10 , 0)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(11 , 0)]] RWStructuredBuffer<LUT> LUT_Buffer;

[[vk::binding(12 , 0)]] RWStructuredBuffer<ReservoirDI> ReSTIRDIBuffer;
[[vk::binding(13 , 0)]] RWStructuredBuffer<ReservoirDI> prevReSTIRDIBuffer;
[[vk::binding(14 , 0)]] RWStructuredBuffer<ReservoirGI> ReSTIRGIBuffer;
[[vk::binding(15 , 0)]] RWStructuredBuffer<ReservoirGI> prevReSTIRGIBuffer;

[[vk::binding(16 , 0)]] TextureCube skybox;
[[vk::binding(17 , 0)]] SamplerState samp;
[[vk::binding(18 , 0)]] Texture2D transmissionLUT;
[[vk::binding(19 , 0)]] Texture1D multiScatter;

[[vk::image_format("rgba16f")]]
[[vk::binding(20 , 0)]] RWTexture2D<float4> diffuseImage;

[[vk::image_format("rgba16")]]
[[vk::binding(21 , 0)]] RWTexture2D<float4> specularImage;

[[vk::image_format("rgba16")]]
[[vk::binding(22 , 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("rgba16")]]
[[vk::binding(23 , 0)]] RWTexture2D<float4> indirectImage;

[[vk::image_format("rgba16")]]
[[vk::binding(24 , 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(25 , 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("r32f")]]
[[vk::binding(26 , 0)]] RWTexture2D<float> roughnessImage;

// the ao map of the material : 1 where there is no map . the DIRECT light is multiplied by it
[[vk::image_format("r32f")]]
[[vk::binding(39 , 0)]] RWTexture2D<float> objectAmbientImage;

[[vk::image_format("r32f")]]
[[vk::binding(27 , 0)]] RWTexture2D<float> metallicImage;

[[vk::image_format("r32f")]]
[[vk::binding(28 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("rg32f")]]
[[vk::binding(29 , 0)]] RWTexture2D<float2> shadowImage;

[[vk::image_format("rg32f")]]
[[vk::binding(30 , 0)]] RWTexture2D<float2> prevShadowImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(31 , 0)]] RWTexture2D<float4> viewImage;

[[vk::image_format("rg32f")]]
[[vk::binding(32 , 0)]] RWTexture2D<float2> motionVector;

[[vk::image_format("r32f")]]
[[vk::binding(33 , 0)]] RWTexture2D<float> pixelCheck;

[[vk::binding(34 ,0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(35 , 0)]] RWStructuredBuffer<uint> tileLightIndices;

// the G buffer of the LAST frame ( the images of the other parity ) : the same formats as the images of this frame
[[vk::image_format("rgba32f")]]
[[vk::binding(36 , 0)]] RWTexture2D<float4> prevPositionImage;

[[vk::image_format("rgba16")]]
[[vk::binding(37 , 0)]] RWTexture2D<float4> prevNormalImage;

[[vk::image_format("r32f")]]
[[vk::binding(38 , 0)]] RWTexture2D<float> prevDepthImage;

[[vk::binding(0 , 2)]] RWStructuredBuffer<lg::LightSource> lightBuffer;
[[vk::binding(1 , 2)]] RWStructuredBuffer<lg::LightCell> lightCells;
[[vk::binding(2 , 2)]] ConstantBuffer<lg::GridCascadeInfo> gridInfo;

// set 1 : the textures of the TextureManager2 ( the same set as in GI_hit.hlsl ) : the hit of a reflection ray reads its material from them
[[vk::binding(0 , 1)]] StructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState typicalSampler;
[[vk::binding(2 , 1)]] Texture2D allTextures[];


[[vk::binding(0 , 3)]] RWStructuredBuffer<RC::ProbeInfo> RCProbeBuffer;
[[vk::binding(1 , 3)]] RWStructuredBuffer<RC::CascadeInfo> RCCascadesBuffer;
[[vk::binding(2 , 3)]] ConstantBuffer<RC::GeneralProbeInfo> RCProbeInfo;
[[vk::image_format("rgba16f")]]
[[vk::binding(3 , 3)]] RWTexture2D<float4> RCImages[];

struct LightData{
    float3 diffuse;
    float3 specular;
    float3 indirect;
    float shadow;
    // the REAL distances ( not weights ) : the denoisers take the radius of the filter from them . diffuse and indirect share the first one
    float diffuseHitDistance;
    float specularHitDistance;
    // the number of the frames that the shadow of the pixel has gathered ( the second channel of the shadow image , 1 .. MAX_SHADOW_FRAME )
    float shadowFrames;
};

struct PixelData{
    float3 position;
    float3 normal;
    float3 albedo;
    float3 viewDir;
    float2 motionVector;
    float depth;
    float metallic;
    float roughness;
    float pixelType;
};

struct [raypayload] RayPayLoad{
    uint depth : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    uint seed : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    PixelData pixelInfo : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    LightData lightInfo : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    bool fromReflection : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
};

struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float depth : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

struct [raypayload] HitRayPayLoad{
    bool   didHit : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 worldPos : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 normal : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 albedo : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float  roughness : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float  metallic : write(caller,closesthit , miss) : read(caller,closesthit , miss);
};

uint bufferIndex(uint2 pixel){
    const uint2 clamped = min(pixel , uint2(generalBuffer.screenWidth - 1u , generalBuffer.screenHeight - 1u));
    return clamped.x + clamped.y * generalBuffer.screenWidth;
}

float3 ComputePixelRayDirection(uint2 pixelCoord, uint2 screenSize, float4x4 invViewProj , float3 playerPosition)
{
    float2 uv = (float2(pixelCoord) + 0.5) / float2(screenSize);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float4 farPoint = mul(invViewProj, float4(ndc, 1.0, 1.0));
    farPoint /= farPoint.w;

    return normalize(farPoint.xyz - playerPosition);   
}

float calcMultiScatterFactor(float3 F0 , float roughness){
    // average of the three channels, not their sum
    const float F0_avr = (F0.x + F0.y + F0.z) / 3.0f;
    const float Ess = multiScatter.SampleLevel(samp , roughness, 0).x;
    return 1.0f + F0_avr * (1.0f / max(Ess , 0.001f) - 1.0f);
}

void findLightIndices(uint2 currentPixel , out uint firstLight , out uint lightCount){
    firstLight = 0;
    lightCount = 0;

    uint2 tileCoord = currentPixel / forwardInfo.TILE_SIZE;
    uint tileIndex = tileCoord.x + tileCoord.y * forwardInfo.TILE_PER_ROW;

    // forward.hlsl's InterlockedAdd keeps counting past MAX_LIGHTS (it only stops WRITING there)
    lightCount = min(lightCountPerTile[tileIndex] , (uint)MAX_LIGHTS);
    firstLight = tileIndex * MAX_LIGHTS;
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

void findLightIndices(const float3 P , out uint lightIndices[lg::MAX_LIGHT_PER_CELL] ,out uint lightCount){
    lightCount = 0;
    const int gridID = findGridSlot(P);
    if(gridID == NON_VALID_SLOT) return;

    const lg::LightCell cell = lightCells[gridID];
    // the grid shader keeps counting past MAX_LIGHT_PER_CELL ( it only stops WRITING there )
    lightCount = min(cell.totalLightCount , (uint)lg::MAX_LIGHT_PER_CELL);
    lightIndices = cell.lightIndices;
}

// the attenuation of a point light : the physical falloff times a window that reaches exactly 0 at lightRadius .
// without the window a light shone on EVERY pixel of the tiles it was listed in , far beyond its radius : the lit and the dark parts met
// on the edges of the Forward+ tiles ( hard vertical and horizontal borders ) and on the sphere of the radius
float lightAttenuation(lg::LightSource light , float lightDistance){
    const float falloff = 1.0f / (1.0f + light.Linear * lightDistance + light.Quadratic * lightDistance * lightDistance);
    const float ratio = saturate(lightDistance / max(light.radius , 1e-4f));
    float window = saturate(1.0f - ratio * ratio * ratio * ratio);
    window *= window;
    return falloff * window;
}


// ---------------------------------------------------------------- the sun disc
static const float SUN_ANGULAR_RADIUS = 0.00465f;
static const float SUN_DISC_BRIGHTNESS = 100.0f;
static const float SUN_DISC_MAX = 60000.0f;

float3 sunTransmittanceToCamera(){
    const float u = skyBuffer.sunCosZenith * 0.5f + 0.5f;
    const float v = saturate(skyBuffer.cameraHeight / (ATMOSPHERE_RADIUS - PLANET_RADIUS));
    return transmissionLUT.SampleLevel(samp , float2(u , v) , 0).rgb;
}

float3 AddSunDisc(float3 skyColor , float3 rayDir){
    const float lightLength = length(sunBuffer.direction);
    // the light is the direction where the light goes : the sun is on the other side
    if(lightLength < 1e-5f) return skyColor;
    const float3 toSun = -sunBuffer.direction / lightLength;

    // nothing under the horizon
    if(rayDir.y < 0.0f) return skyColor;

    // the angle of one pixel from the projection ( projOnly[1][1] is 1 / tan( fov / 2 ) )
    const float pixelAngle = 2.0f / (max(generalBuffer.projOnly[1][1] , 1e-3f) * float(max(generalBuffer.screenHeight , 1u)));
    const float cosAngle = dot(rayDir , toSun);
    if(cosAngle < cos(SUN_ANGULAR_RADIUS + pixelAngle)) return skyColor;

    const float angle = acos(clamp(cosAngle , -1.0f , 1.0f));
    const float coverage = saturate((SUN_ANGULAR_RADIUS + 0.5f * pixelAngle - angle) / pixelAngle);

    const float3 disc = min(sunBuffer.diffuse * SUN_DISC_BRIGHTNESS * sunTransmittanceToCamera() , float3(SUN_DISC_MAX , SUN_DISC_MAX , SUN_DISC_MAX));
    return skyColor + disc * coverage;
}

// ReSTIR DI is switched by features.ReSTIR ( ImGui ) : there is nothing to recompile

float targetPdf_DI(uint lightIndex , GeneralObjectInfo info){
    const lg::LightSource candidate = lightBuffer[lightIndex];
    float3 L = candidate.position - info.P;
    const float d = length(L);
    L /= max(d , 1e-4f);

    const float attenuation = lightAttenuation(candidate , d);
    float3 diffuse;
    float3 specular;

    computeLight(info , L , diffuse ,specular);
    const float3 brdfValue = (diffuse + specular) * max(dot(info.N , L) , 0.0001f);
    return luminance(candidate.diffuse * brdfValue * attenuation);
}



// shadow calc
float calculateShadows(float3 position , float3 direction , float maxDistance){
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    ShadowRayPayLoad shadow;
    shadow.inShadow = true;
    shadow.depth = maxDistance;
    TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH , 0xFF , RT_HITGROUP_SHADOW , 0 , RT_MISS_SHADOW , ray, shadow);
    const float data = (shadow.inShadow )? 0.0f  : 1.0f;
    return data;
}

// this funcntion also 
float calculateShadowAtt(float3 position , float3 direction , float maxDistance){
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    ShadowRayPayLoad shadow;
    shadow.inShadow = true;
    shadow.depth = maxDistance;
    // no ACCEPT_FIRST_HIT : the depth has to be the one of the NEAREST blocker ( the first accepted hit can be a farther one ) , as in calculateSunShadow
    TraceRay(TLAS, RAY_FLAG_NONE , 0xFF , RT_HITGROUP_SHADOW , 0 , RT_MISS_SHADOW , ray, shadow);
    float data = 1.0f;
    if(!shadow.inShadow) return data;
    // VISIBILITY of the lamp along this ray : r = depth / distance to the lamp . a blocker at the pixel ( r -> 0 ) gives 0 ( the full shadow ) , a blocker at the lamp
    // ( r -> 1 ) gives 1 ( no shadow ) . it was 1 - r : the contact shadow was the brightest one and the far one the darkest
    const float r = saturate(shadow.depth / max(maxDistance , 1e-20f));
    return r;
}

// the shadow of the sun with a fade : how much of the sun reaches the pixel along this ray . 1 - nothing covers it , 0 - covered at once . the shadow gets weaker
// with the distance from the pixel to what covers it ( the nearest hit along the ray to the sun , it is the depth that shadowHit.hlsl writes ) : it is fully gone at
// fadeDistance ( smoothstep , features.sunShadowFade , in meters ) . fadeDistance 0 - no fade , the same as calculateShadows
float calculateSunShadow(float3 position , float3 direction , float maxDistance , float fadeDistance){
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    ShadowRayPayLoad shadow;
    shadow.inShadow = true;
    shadow.depth = maxDistance;

    if(fadeDistance <= 0.0f){
        TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH , 0xFF , RT_HITGROUP_SHADOW , 0 , RT_MISS_SHADOW , ray, shadow);
        return shadow.inShadow ? 0.0f : 1.0f;
    }

    // no ACCEPT_FIRST_HIT : the closest hit shader runs for the NEAREST accepted hit ( the lamp bodies are ignored by the any hit ) , and its depth is what is needed
    TraceRay(TLAS, RAY_FLAG_NONE , 0xFF , RT_HITGROUP_SHADOW , 0 , RT_MISS_SHADOW , ray, shadow);
    if(!shadow.inShadow) return 1.0f;
    return smoothstep(0.0f , fadeDistance , shadow.depth);
}

float3 sampleCone(const float3 axis , const float cosAlpha , inout uint seed){
    const float phi = 2.0f * PI * rand(seed);
    const float rand2 = rand(seed);

    const float cosTheta = lerp(cosAlpha , 1.0f , rand2);
    const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    float3 tangent , bitangent;
    buildONB(axis , tangent , bitangent);
    return localDir.x * tangent + localDir.y * bitangent + localDir.z * axis;
}

float3 randomInLightRadius(inout uint seed  , lg::LightSource light , const float3 L){
    const float u = rand(seed) , v = rand(seed);
    const float z = 1.0f - 2 * u;
    const float phi = 2.0f * PI * v;
    const float r = sqrt(saturate(1.0f - z * z));
    const float x = cos(phi) * r;
    const float y = sin(phi) * r;
    const float R = light.sourceRadius;
    float3 offset = R * float3(x ,y ,z);
    // L goes FROM the point TO the lamp , the visible half of the sphere is the one that looks back at the point : the offsets that look the way of L ( dot > 0 )
    // are on the far side of the lamp , they are turned over ( it was dot < 0 : the far half was chosen )
    const float facing = dot(offset , L);
    if(facing > 0.0f) offset = -offset;
    return light.position + offset;
}

float3 randomInSunRadius(inout uint seed  , SunSource light , const float3 L){
    const float u = rand(seed) , v = rand(seed);
    const float z = 1.0f - 2 * u;
    const float phi = 2.0f * PI * v;
    const float r = sqrt(saturate(1.0f - z * z));
    const float x = cos(phi) * r;
    const float y = sin(phi) * r;
    const float R = light.sourceRadius;
    float3 offset = R * float3(x ,y ,z);
    // L goes FROM the point TO the lamp , the visible half of the sphere is the one that looks back at the point : the offsets that look the way of L ( dot > 0 )
    // are on the far side of the lamp , they are turned over ( it was dot < 0 : the far half was chosen )
    const float facing = dot(offset , L);
    if(facing > 0.0f) offset = -offset;
    // the sun is at infinity : its "sphere" is a disc of the angular radius R on the unit sphere of directions , so the result is a DIRECTION ( the centre L plus the offset , normalized ) ,
    // not the offset alone ( it was a vector of length ~0.005 : dot(N , L) and the light went to 0 )
    return normalize(normalize(L) + offset);
}

float calculateShadowInSphere(inout uint seed , lg::LightSource light , const float3 P , const float3 N){
    float3 L = light.position - P;
    const float d = length(L);
    L = normalize(L);
    const float3 randomPos = randomInLightRadius(seed , light , L);
    // the ray ends at the chosen point of the sphere , not at its centre ( offsetRay wants the position of the camera as the third argument , not a distance )
    const float3 toSample = randomPos - P;
    const float sampleDistance = length(toSample);
    const float3 dir = toSample / max(sampleDistance , 1e-4f);
    return calculateShadowAtt(offsetRay(P , N , generalBuffer.playerPosition) , dir , sampleDistance);
}

// the shadow of a lamp for the INDIRECT light ( the lamps that light the hits of the GI , Radiance Cascade , the reflections and the bounces of the path tracer ) : the ray goes to the
// CENTRE of the lamp , not to a random point of its sphere . with a big sphere the random point is on the other side of a wall , and the light of a lamp under the floor leaked through
float calculateShadowToCenter(lg::LightSource light , const float3 P , const float3 N){
    const float3 toLight = light.position - P;
    const float distance = length(toLight);
    const float3 dir = toLight / max(distance , 1e-4f);
    // the FULL shadow ( 0 or 1 ) : calculateShadowAtt gives the visibility r = depth / distance , a wall near the lamp then let a part of the light through
    return calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , dir , distance);
}



float3 stretchView(const float3 V , const float a){
    return normalize(float3(a  * V.x , V.y * a , V.z));
}

GeneralObjectInfo infoFromGBuffer(uint2 pixel){
    GeneralObjectInfo info;
    info.P = positionImage[pixel].xyz;
    info.N = normalImage[pixel].xyz;
    info.V = normalize(generalBuffer.playerPosition - info.P);
    info.albedo = albedoImage[pixel].xyz;
    info.roughness = roughnessImage[pixel];
    info.metallic = metallicImage[pixel];
    return info;
}

float3 sampleSkybox(const float3 N){
    float3 result = 0.0f;
    // 0 samples : no sky light ( and no division by 0 below )
    if(features.skyboxSampleCount == 0u) return result;
    [loop]
    for(uint i = 0 ; i  < features.skyboxSampleCount ; i++){
        const float3 direction = randomInHemisphere(N);
        result += skybox.SampleLevel(samp , direction , 0).xyz;
    }
    result /= float(features.skyboxSampleCount);
    return result;
}

namespace ReSTIRGI{

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

    // indirect GI
    // ReSTIR GI

    bool isSpecularMode(const float roughness , const float metallic){ return (metallic >= 0.7f || roughness < 0.4f); }

    float3 evalFcos(const GeneralObjectInfo info , const bool specularMode , const float3 L){
        if(specularMode){
            const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
            return evalBRDF(info.N , info.V , L , info.roughness , F0) * calcMultiScatterFactor(F0 , info.roughness);
        }
        return evalDiffuse(info.albedo , info.N , L , info.metallic);
    }

    // next task is to fix the sun shadows
    float3 computeLightAt(HitRayPayLoad info, inout uint seed, const float roughness , const float xMetallic, const float3 x1Pos){
        // in order to literally minimaze the atmosphere effect , we actually need to lower the chances of it to appear * 
        if(!info.didHit){
            return 0.0f;
        }
        uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
        uint2 pixel = DispatchRaysIndex().xy;

        findLightIndices(info.worldPos , lightIndices , lightCount);

        GeneralObjectInfo data = (GeneralObjectInfo)0;
        data.albedo = info.albedo;
        data.metallic = info.metallic;
        data.roughness = info.roughness;
        data.N = info.normal;
        data.P = info.worldPos;
        
        const float3 saveAlbedo = max(info.albedo , 1e-20);

        float3 result = 0.0f ;
        if(lightCount){
            const uint id = (uint)(rand(seed) * lightCount);
            lg::LightSource light = lightBuffer[lightIndices[id]];
            float3 diffuse, specular;

            const float3 V = normalize(x1Pos - info.worldPos);
            data.V = V;
            float3 L = light.position - data.P;
            L = normalize(L);
            // the centre of the lamp , not a random point of its sphere ( the indirect light : a big sphere reached through the walls )
            float3 lightPos = light.position;
            float3 realL = lightPos - info.worldPos;
            const float dist = length(realL);
            realL = normalize(realL);
            
            computeLight(data ,realL , diffuse , specular );

            const float NdotL = saturate(dot(data.N , realL));  
            float shadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , dist);
            float window = lightAttenuation(light, dist);
            diffuse /= saveAlbedo;
            result = NdotL * light.diffuse * (diffuse) * shadow  * window * lightCount;
        }else{
            result =  0.01f;
        }

        if(sunBuffer.intensity > 0.0f){
            float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
            // now we also have to check if the shadow actually hits it , since if not, there is no point at computing the light , since its also the calcualtion
            const float3 L = randomInSunRadius(seed , sunBuffer , -sunBuffer.direction);
            const float sunShadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , 1000.0f);
            computeLight(data , L , sunDiffuse , sunSpecular);
            sunDiffuse /= saveAlbedo;
            result += (sunDiffuse) *  clamp(dot(data.N , L),  0.0f , 1.0f) * sunBuffer.diffuse  * sunShadow;
        }

        {
            const float3 V2 = normalize(x1Pos - info.worldPos);
            const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
            const float NdotV = max(dot(info.normal , V2) , 0.0001f);
            const float3 F = F_fresnel(F0 , NdotV);
            const float3 kD = (1.0f - F) * (1.0f - info.metallic);
            result += sampleSkybox(info.normal) * kD;
        }

        result *= saveAlbedo;

        return result;
    }

    ReservoirGI generateReservoirGI(GeneralObjectInfo info , const uint2 pixel ,inout uint seed){
        // the DIFFUSE lobe only : the reflection is a separate ray ( traceRayReflection ) , the two lobes are summed in processPixel
        ReservoirGI res = (ReservoirGI)0;
        float3 L ; float pdf;
        sampleDiffuseDirection(float2(randNoise() , randNoise()) , info.N ,  L , pdf);

        if(pdf <= 0.0f) return res;
        RayDesc ray;
        ray.Origin = offsetRay(info.P , info.N , generalBuffer.playerPosition);
        ray.Direction = L;
        ray.TMax = 1000.0f;
        ray.TMin = 0.0001f;

        HitRayPayLoad load = (HitRayPayLoad)0;

        TraceRay(TLAS , RAY_FLAG_NONE , 0xFF , RT_HITGROUP_HITINFO , 0 , RT_MISS_HITINFO , ray , load);

        if(load.didHit){

            const float3 brdf = computeLightAt(load,  seed, info.roughness , info.metallic , info.P);

            res.radiance = brdf;

            const float3 fcos = evalDiffuse(info.albedo , info.N , L , info.metallic);
            
            const float  p_hat = luminance(res.radiance * fcos);
            res.weightSum = p_hat / pdf;
            res.W = 1.0f / pdf;    

        }else{
        
            const float3 F0 = lerp(info.albedo , float3(0.04, 0.04 , 0.04) , (1.0f - info.metallic));
            const float3 V = normalize(generalBuffer.playerPosition - info.P);
            const float NdotV = max(dot(info.N , V) , 0.0001f);
            const float3 F = F_fresnel(F0 , NdotV);
            // (1 - metallic) and the albedo are NOT here : evalDiffuse ( processPixel , and p_hat below ) has both of them , a second one would be a square
            const float3 kD = (1.0f - F);
            res.radiance = sampleSkybox(info.N) * kD;
            // the same weights as a sample that hit something ( the direction L was chosen with the pdf of the cosine ) : radiance * evalDiffuse * W is
            // radiance * albedo then . W = 1 gave radiance * albedo * cos / PI , 6 times less
            const float3 fcos = evalDiffuse(info.albedo , info.N , L , info.metallic);
            const float  p_hat = luminance(res.radiance * fcos);
            res.weightSum = p_hat / pdf;
            res.W = 1.0f / pdf;
        }

        res.sampleNormal = load.normal;
        res.samplePos = load.worldPos;
        res.roughness = load.roughness;
        res.M = 1;

        return res;
    }

    // the history of GI is the one of the same SURFACE : the position of the pixel of the last frame has to be about the same as the position of this one . without it the
    // reservoir of whatever was behind ( or in front of ) the point in the last frame was reused : the light of another surface . w == 0 marks a sky texel
    bool historyMatchesPosition(const float3 P , const uint2 prevPixel){
        const float4 prevPosition = prevPositionImage[prevPixel];
        if(prevPosition.w == 0.0f) return false;
        // the same tolerance as the check of the shadow and of Radiance Cascade ( RC::checkDisocclusion )
        return length(P - prevPosition.xyz) <= 0.05f;
    }

    ReservoirGI temporalReuse_GI( inout uint seed ,ReservoirGI current , uint2 prevCoord , GeneralObjectInfo info){
        ReservoirGI r = (ReservoirGI)0;

        ReservoirGI prevGI = prevReSTIRGIBuffer[bufferIndex(prevCoord)];

        const uint M_CAP = 30u;
        prevGI.M = min(prevGI.M , M_CAP * max(current.M , 1u));

        {
            // the target function is THE SAME one as in generateReservoirGI : the radiance times f * cos of the diffuse lobe
            const float3 L_current = normalize(current.samplePos - info.P);
            const float p_hat_current = luminance(current.radiance * evalDiffuse(info.albedo , info.N , L_current , info.metallic));
            const float weight_current = p_hat_current * current.W * float(current.M);
            r.update(current , weight_current , current.M , rand(seed));
        }

        {
            const float3 L_prev = normalize(prevGI.samplePos - info.P);
            const float p_hat_prev = luminance(prevGI.radiance * evalDiffuse(info.albedo , info.N , L_prev , info.metallic));
            const float weight_prev = p_hat_prev * prevGI.W * float(prevGI.M);
            r.update(prevGI , weight_prev , prevGI.M , rand(seed));
        }

        const float3 L_chosen = normalize(r.samplePos - info.P);
        const float p_hat_chosen = luminance(r.radiance * evalDiffuse(info.albedo , info.N , L_chosen , info.metallic));
        r.W = (p_hat_chosen > 1e-6f && r.M > 0u) ? r.weightSum / (float(r.M) * p_hat_chosen) : 0.0f;
        return r;
    }

    ReservoirGI combineGIJacobian(ReservoirGI a, ReservoirGI b , GeneralObjectInfo main, const float3 nP , inout uint seed){
        ReservoirGI output = (ReservoirGI)0;

        float3 Q = b.samplePos;
        float3 Q_normal = b.sampleNormal;

        float3 dirFromQ = Q - nP;
        float distN = length(dirFromQ);
        float cosN = abs(dot(normalize(-dirFromQ), Q_normal));
        float3 dirFromMine = Q - main.P;
        float distMine = length(dirFromMine);
        float cosMine = abs(dot(normalize(-dirFromMine),  Q_normal));
        float jacobian = (cosMine * distN * distN) / max(cosN * distMine * distMine , 1e-20);

        jacobian = clamp(jacobian , 0.0f , 10.0f);

        output = b;
        // the neighbour counts for at most 20 samples : without the cap M grew without limit ( every frame it takes the M of 8 neighbours that took the M of their neighbours ) ,
        // overflowed , and the weights became inf / NaN . the NaN stays in the history of the denoiser forever : the whole picture went black
        output.M = min(b.M , 20u);
        output.W *= jacobian;

        float3 L = normalize(Q - main.P);
        float p_hat = luminance(output.radiance * evalDiffuse(main.albedo , main.N , L , main.metallic));
        float weight = p_hat * output.W * float(output.M);

        a.update(output , weight , output.M , rand(seed));
        {
            // p_hat of the CHOSEN sample ( a.radiance , a.samplePos ) : it used output.radiance ( the neighbour's ) , which is another sample when the reservoir kept its own
            float3 L = normalize(a.samplePos - main.P);
            float p_hat = luminance(a.radiance * evalDiffuse(main.albedo , main.N , L , main.metallic));
            a.W = (p_hat > 1e-6f) ? a.weightSum / (float(a.M) * p_hat) : 0.0f;
        }
        return a;
    }

    uint pickRandomNumber(inout uint seed , uint totalCount){
        return uint(float(totalCount) * rand(seed));
    }

    ReservoirGI spatialReuse(ReservoirGI a , GeneralObjectInfo main , inout uint seed){

        ReservoirGI current = a;

        uint2 id = DispatchRaysIndex().xy;
        uint2 size = DispatchRaysDimensions().xy;
        [unroll]
        for(uint i = 0 ; i < 8 ; i++){
            const uint index = pickRandomNumber(seed , 16);
            // PoissonDisk is in the unit disc : in pixels it is the radius times it ( without the radius int2( offset ) is 0 or +-1 : the neighbour was the pixel itself )
            float2 offset = PoissonDisk[index];
            int2 pixel = id + int2(offset);
            if(any(pixel >= size) || any(pixel < 0)) continue;
            float3 P = positionImage[pixel].xyz;
            float3 N = normalImage[pixel].xyz;


            if(dot(N , main.N) < 0.9f) continue;
            if(abs(length(P - main.P)) > 2.0f) continue;
            {
                float3 dir = P - a.samplePos;
                const float d = length(dir);
                dir = normalize(dir);

                const float occlusionCheck = calculateShadows(offsetRay(a.samplePos , a.sampleNormal , generalBuffer.playerPosition) , dir , d);
                if(occlusionCheck != 1.0f) continue;
            }
            ReservoirGI n = prevReSTIRGIBuffer[bufferIndex(pixel)];
            if(n.M == 0u || n.W <= 0.0f) continue;
            {
                float3 dir = n.samplePos - a.samplePos;
                const float d = length(dir);
                dir = normalize(dir);

                const float occlusionCheck = calculateShadows(offsetRay(a.samplePos , a.sampleNormal , generalBuffer.playerPosition) , dir , d);
                if(occlusionCheck != 1.0f) continue;
            }

            current = combineGIJacobian(current , n , main, P , seed);
        }

        // the total M is capped too ( the weights are scaled together with it , W does not change ) : the reservoir goes to the buffer of the next frame
        const uint M_CAP = 30u;
        if(current.M > M_CAP){
            current.weightSum *= float(M_CAP) / float(current.M);
            current.M = M_CAP;
        }

        return current;
    }

    ReservoirGI ReSTIR_GI_generation(inout uint seed , GeneralObjectInfo info){
        ReservoirGI res = (ReservoirGI)0;
        if(!features.ReSTIRGI){
            res.W = 1.0f;
            res.radiance = 1.0f;
            return res;
        }

        uint2 pixel = DispatchRaysIndex().xy;

        res = generateReservoirGI(info , pixel  , seed);

        if(features.ReSTIRGI_spatial){
            // the result has to be taken : spatialReuse works on a copy ( it returns the new reservoir ) , without this the call changed nothing
            res = spatialReuse(res , info , seed);
        }
        if(features.ReSTIRGI_temporal){
            uint2 size = DispatchRaysDimensions().xy;
            float4 clipPos = mul(generalBuffer.prevViewProj , float4(info.P , 1.0f));
            float3 pos = clipPos.xyz / clipPos.w;
            float2 uv = pos.xy * 0.5f + 0.5f;
            uv.y = 1.0f - uv.y;
            // the check is made BEFORE the conversion to uint : a negative float does not become a pixel outside of the screen , and an unsigned
            // value is never < 0 ( the old check was always true )
            const bool validUV = (uv.x >= 0.0f && uv.x < 1.0f && uv.y >= 0.0f && uv.y < 1.0f);
            if(validUV){
                const uint2 prevPixel = min(uint2(uv * float2(size)) , size - 1u);
                // only the history of the same surface is reused ( the position of the last frame is close to this one )
                if(historyMatchesPosition(info.P , prevPixel)){
                    res = temporalReuse_GI(seed , res , prevPixel ,info);
                }
            }
        }



        return res;
    }
}

static const float INDIRECT_FALLOFF_DISTANCE = 8.0f;

float indirectDistanceWeight(float hitDistance){
    const float ratio = hitDistance / INDIRECT_FALLOFF_DISTANCE;
    return 1.0f / (1.0f + ratio * ratio);
}
static const float MAX_LIGHT_CONE_SIN = 0.05f;

void finalizeReservoir_DI(inout ReservoirDI r , GeneralObjectInfo info){
    // an undefined light has no p_hat : lightBuffer must not be read with this index
    if(r.chosenIndex == 9999u){
        r.W = 0.0f;
        return;
    }
    const float p_hat = targetPdf_DI(r.chosenIndex , info);
    r.W = (p_hat > 1e-6f && r.M > 0u) ? r.weightSum / (float(r.M) * p_hat) : 0.0f;
}

void generateReservoir_DI(inout ReservoirDI res , inout uint seed , GeneralObjectInfo info, uint lightIndices[lg::MAX_LIGHT_PER_CELL] , uint lightCount){
    res.W = 0.0f;
    res.M = 0;
    res.weightSum = 0.0f;
    res.chosenIndex = 9999u;   // undefined : no light is chosen
    res.diffuseColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
    res.specularColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
    if(lightCount == 0u) return;

    float bestPdf = 0.0f;
    for(uint i = 0 ; i < lightCount ; i++){
        const uint lightIndex = lightIndices[i];
        const lg::LightSource light = lightBuffer[lightIndex];

        float3 toLight = light.position - info.P;
        const float lightDistance = length(toLight);
        toLight /= max(lightDistance , 1e-4f);
        const float3 lightPos = randomInLightRadius(seed , light , toLight);
        float3 dir = lightPos - info.P;
        const float d = length(dir);
        dir = normalize(dir);
        float3 candidateDiffuse;
        float3 candidateSpecular;
        const float shadow = calculateShadowInSphere(seed  , light , info.P , info.N);
        computeLight(info , dir , candidateDiffuse , candidateSpecular);

        float3 radiance = 0.0f;
        if(shadow == 1.0f){
            radiance = light.diffuse * lightAttenuation(light , d) * max(dot(info.N , dir) , 0.0001f);
        }else{
            radiance = info.albedo * 0.001f;
        }
        candidateDiffuse *= radiance;
        candidateSpecular *= radiance;

        res.diffuseColor.xyz += candidateDiffuse;
        res.specularColor.xyz += candidateSpecular;

        const float p_hat = luminance(candidateDiffuse + candidateSpecular);
        res.weightSum += p_hat;
        // a light wins only with p_hat > 0
        if(p_hat > bestPdf){
            bestPdf = p_hat;
            res.chosenIndex = lightIndex;
        }
    }

    // nothing reaches the pixel : no light is chosen ( M = 0 , W = 0 ) , and finalizeReservoir_DI is not called with an undefined index
    if(res.chosenIndex == 9999u) return;

    res.M = 1u;
    finalizeReservoir_DI(res , info);
}

ReservoirDI combineReservoir_DI(ReservoirDI a , ReservoirDI b , inout uint seed){
    ReservoirDI result;
    if(a.chosenIndex == 9999) return b;
    if(b.chosenIndex == 9999) return a;

    result.M = a.M + b.M;
    result.weightSum = a.weightSum + b.weightSum;
    result.chosenIndex = a.chosenIndex;
    result.W = 0.0f;   // recomputed by finalizeReservoir_DI

    if(result.weightSum > 0.0f && rand(seed) * result.weightSum < b.weightSum){
        result.chosenIndex = b.chosenIndex;
    }
    return result;
}

void temporalReuse_DI(inout ReservoirDI w , inout uint seed , uint2 prevPixel, GeneralObjectInfo info){
    const ReservoirDI prev = prevReSTIRDIBuffer[bufferIndex(prevPixel)];
    if(prev.M == 0u) return;

    const uint M_clamped = min(prev.M , 20u * max(w.M , 1u));

    ReservoirDI pRes;
    pRes.M = M_clamped;
    pRes.chosenIndex = prev.chosenIndex;
    if(prev.chosenIndex == 9999) return;
    // canonical ReSTIR weight of a reused sample: p_hat_here * W * M
    pRes.weightSum = prev.W * float(M_clamped) * targetPdf_DI(prev.chosenIndex , info);
    pRes.W = 0.0f;

    w = combineReservoir_DI(w , pRes , seed);
    finalizeReservoir_DI(w ,info);
}

void spatialReuse_DI(inout ReservoirDI r , inout uint seed , uint2 currentPixel, GeneralObjectInfo info){
    const uint numSpatialSamples = 5u;
    const float spatialRadius = 30.0f;   // in pixels

    for(uint s = 0u; s < numSpatialSamples; s++){
        const float rand1 = rand(seed);
        const float rand2 = rand(seed);
        const float phi = 2.0f * PI * rand2;

        const float radius = sqrt(rand1) * spatialRadius;
        const int2 neighborPixel = int2(currentPixel) + int2(cos(phi) * radius , sin(phi) * radius);
        // signed compare - casting to uint first made the "< 0" test dead
        if(neighborPixel.x < 0 || neighborPixel.y < 0 ||
           neighborPixel.x >= (int)generalBuffer.screenWidth || neighborPixel.y >= (int)generalBuffer.screenHeight){
            continue;
        }

        const ReservoirDI stored = prevReSTIRDIBuffer[bufferIndex((uint2)neighborPixel)];
        if(stored.M == 0u || stored.W <= 0.0f) continue;
        if(stored.chosenIndex == 9999) continue;
        // Geometric rejection used to read normalOutputImage/depthOutputImage of
        // the neighbour, which other invocations of this very dispatch are still
        // writing - a data race. Re-evaluating the target function at THIS pixel
        // rejects unusable neighbours without touching the shared G-buffer.
        const float p_hat = targetPdf_DI(stored.chosenIndex , info);
        if(p_hat <= 0.0f) continue;

        ReservoirDI neighborRes;
        neighborRes.chosenIndex = stored.chosenIndex;
        neighborRes.M = stored.M;
        neighborRes.W = 0.0f;
        // the stored.W factor was missing here while temporalReuse_DI had it
        neighborRes.weightSum = stored.W * float(stored.M) * p_hat;

        r = combineReservoir_DI(r , neighborRes , seed);
        finalizeReservoir_DI(r ,info);
    }
}

ReservoirDI ReSTIR_DI_generation(uint2 currentPixel , float2 prevUV , inout uint seed , GeneralObjectInfo info,  uint lightIndices[lg::MAX_LIGHT_PER_CELL] , uint lightCount){
    const bool validUV = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    const uint2 prevCoords = (uint2)clamp(prevUV * float2(generalBuffer.screenWidth , generalBuffer.screenHeight) ,
                                          float2(0.0f , 0.0f) ,
                                          float2(generalBuffer.screenWidth - 1u , generalBuffer.screenHeight - 1u));

    // every light is visited in generateReservoir_DI , so there is nothing to collect from the last frame or from the neighbours . ( the reuse picks between
    // the reservoirs by a random number : it would bring the flashes back )
    ReservoirDI r;
    generateReservoir_DI(r , seed , info , lightIndices , lightCount);
    const uint M_CAP = 500u;
    if(r.M > M_CAP){
        r.weightSum *= float(M_CAP) / float(r.M);
        r.M = M_CAP;
    }
    return r;
}

float3 traceRayReflection(const float3 direction , const float3 position , out float hitDistance){

    RayPayLoad load = (RayPayLoad)0;
    load.fromReflection = true;

    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = 1000.0f;
    ray.TMin = 0.001f;

    TraceRay(TLAS , RAY_FLAG_NONE , 0xFF , RT_HITGROUP_GENERAL , 0 , RT_MISS_GENERAL , ray , load);

    hitDistance = min(load.lightInfo.specularHitDistance , 900u);
    if(load.pixelInfo.pixelType == SKYBOX_PIXEL){
        return load.lightInfo.diffuse;
    }

    // the diffuse of the hit is demodulated ( divided by max( albedo , 0.02 ) ) and holds the indirect light too : the albedo brings it back , the indirect
    // light is taken out of it and put back without the shadow ( the shadow belongs to the direct light only , the indirect light has its own visibility inside )
    const float3 indirect = load.lightInfo.indirect;
    const float3 directDiffuse = max(load.lightInfo.diffuse * max(load.pixelInfo.albedo , 0.02f), 0.0f);
    return (directDiffuse + load.lightInfo.specular) * load.lightInfo.shadow + indirect;
}

// Cascade radiance
namespace RC
{
    // the place of the world point P in the GRID OF THE PROBES of the cascade 0 , as a float : the probes are attached to the tiles of the SCREEN ( not to the world ) , so the
    // point is projected to the screen first . the probe i stands in the middle of its tile , so its centre is at the integer i in these units :
    //   floor( probePos )  - the probe on the left / top of the point , the next one is + 1 , and  probePos - floor( probePos )  is the weight of the next one ( the bilinear )
    // returns false if there is no probe for the point : it is behind the camera , or outside of the screen ( the probes of the screen know nothing about it )
    // it does not look at the depth : a point hidden behind something else gets the place of the pixel in front of it , the caller compares the depth / the normal if it matters
    bool findProbePosition(const float3 P , out float2 probePos)
    {
        probePos = float2(0.0f , 0.0f);

        const float4 clipPos = mul(generalBuffer.viewProj , float4(P , 1.0f));
        if(clipPos.w <= 0.0f) return false;

        float2 uv = (clipPos.xy / clipPos.w) * 0.5f + 0.5f;
        // the same flip as everywhere : the pixel ( 0 , 0 ) is the top left corner , the NDC y points up
        uv.y = 1.0f - uv.y;
        if(any(uv < 0.0f) || any(uv >= 1.0f)) return false;

        const RC::CascadeInfo cascade = RCCascadesBuffer[0];
        const float2 pixel = uv * float2(generalBuffer.screenWidth , generalBuffer.screenHeight);
        probePos = pixel / cascade.PROBE_RESOLUTION - 0.5f;
        return true;
    }

    // the probe of the cascade 0 that is the NEAREST to P : the coordinate ( the column and the row of the tile ) , not an index . int2( -1 , -1 ) if there is no probe
    // the nearest one is the tile that the point falls into , which is the rounding of probePos
    int2 findProbe(const float3 P)
    {
        float2 probePos;
        if(!findProbePosition(P , probePos)) return int2(-1 , -1);

        const RC::CascadeInfo cascade = RCCascadesBuffer[0];
        // the last row / column of the tiles is cut by the edge of the screen , but a probe is there : the number of the probes is what the image holds
        const int2 probeCount = int2(cascade.CASCADE_X , cascade.CASCADE_Y) / int(cascade.BUILD_SIZE);
        return clamp(int2(floor(probePos + 0.5f)) , int2(0 , 0) , probeCount - 1);
    }

    // hitDistance : the distance of the hit that the probes see in the direction ( meters ) , for the radius of the filter of the denoiser : the mean of the distances of the
    // hits ( the sky : 1000 ) with the same weights as the colour . it is 1000 when there is no probe
    float3 processRC(const float3 dir , const float3 P , out float hitDistance , const bool currentImage)
    {
        hitDistance = 1000.0f;
        float2 probePos;
        if(!findProbePosition(P , probePos))
        {
            return sampleSkybox(dir);
        }

        const RC::CascadeInfo c = RCCascadesBuffer[0];
        const int buildSize = int(c.BUILD_SIZE);
        const int2 probeCount = int2(c.CASCADE_X , c.CASCADE_Y) / buildSize;

        // the PROBE : the base is the probe on the left / top of the point , the fraction is how far the point is from it
        const int2 probeBase = int2(floor(probePos));
        const float2 probeFrac = probePos - float2(probeBase);

        // the DIRECTION : the same for the block of every probe
        const float2 texelPos = octahedralEncode(dir) * float(buildSize) - 0.5f;
        const int2 texelBase = int2(floor(texelPos));
        const float2 texelFrac = texelPos - float2(texelBase);

        // N = 1 : the offsets 0 and 1 from the base ( the 2 x 2 of the bilinear ) , a bigger N gives a wider window with the weights of the far ones smaller
        const int PROBE_SAMPLE_COUNT = 1;
        const int BUILD_SAMPLE_COUNT = 1;

        float weightSum = 0.0f;
        float3 result = 0.0f;
        // the alpha of the images is packed ( RC::packHit ) : the samples are unpacked first , the hits are counted by their weights
        float hitWeight = 0.0f;
        float hitDistanceSum = 0.0f;
        float missDistanceSum = 0.0f;

        [unroll]
        for(int probeY = 1 - PROBE_SAMPLE_COUNT ; probeY <= PROBE_SAMPLE_COUNT ; probeY++)
        {
            [unroll]
            for(int probeX = 1 - PROBE_SAMPLE_COUNT ; probeX <= PROBE_SAMPLE_COUNT ; probeX++)
            {
                const int2 currentProbe = probeBase + int2(probeX , probeY);
                if(any(currentProbe < 0) || any(currentProbe >= probeCount)) continue;

                const uint probeIndex = c.PROBE_OFFSET + uint(currentProbe.y) * c.PROBE_PER_ROW + uint(currentProbe.x);
                if(RCProbeBuffer[probeIndex].validProbe == 0u) continue;

                const float2 probeDistance = abs(float2(probeX , probeY) - probeFrac);
                const float probeWeight = saturate(1.0f - probeDistance.x / float(PROBE_SAMPLE_COUNT)) * saturate(1.0f - probeDistance.y / float(PROBE_SAMPLE_COUNT));
                if(probeWeight <= 0.0f) continue;

                [unroll]
                for(int buildY = 1 - BUILD_SAMPLE_COUNT ; buildY <= BUILD_SAMPLE_COUNT ; buildY++)
                {
                    [unroll]
                    for(int buildX = 1 - BUILD_SAMPLE_COUNT ; buildX <= BUILD_SAMPLE_COUNT ; buildX++)
                    {
                        const int2 currentBuild = texelBase + int2(buildX , buildY);
                        // since we dont want to touch other probes
                        if(any(currentBuild < 0) || any(currentBuild >= buildSize)) continue;

                        const float2 texelDistance = abs(float2(buildX , buildY) - texelFrac);
                        const float texelWeight = saturate(1.0f - texelDistance.x / float(BUILD_SAMPLE_COUNT)) * saturate(1.0f - texelDistance.y / float(BUILD_SAMPLE_COUNT));
                        if(texelWeight <= 0.0f) continue;

                        // the corner of the block of the probe + the texel in it
                        const int2 pixel = currentProbe * buildSize + currentBuild;

                        const float weight = probeWeight * texelWeight;
                        const float4 sampleData = (currentImage ? RCImages[RC::imageIndex(0u , generalBuffer.frameIndex)][uint2(pixel)] : RCImages[RC::prevImageIndex(generalBuffer.frameIndex)][uint2(pixel)]);
                        bool sampleHit;
                        float sampleDistance;
                        RC::unpackHit(sampleData.a , sampleHit , sampleDistance);

                        result += weight * sampleData.xyz;
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

        if(weightSum <= 0.0f) return sampleSkybox(dir);

        // the distance of the hits if there are any , of the misses if there are not
        hitDistance = (hitWeight > 0.0f) ? (hitDistanceSum / hitWeight) : (missDistanceSum / weightSum);
        return result / weightSum;
    }



bool checkDisocclusion(uint2 pixel , uint2 prevPixel){

    const float3 centerPosition = positionImage[pixel].xyz;
    const float3 centerNormal = normalImage[pixel].xyz;

    const float4 prevPositionSample = prevPositionImage[prevPixel];
    const float3 nNormal = prevNormalImage[prevPixel].xyz;   

    // w == 0 marks a sky texel written by the miss shader
    if(prevPositionSample.w == 0.0f) return false;

    const float positionDiff = length(centerPosition - prevPositionSample.xyz);
    if(positionDiff > 0.05f) return false;

    const float normalDot = dot(centerNormal , nNormal);
    if(normalDot < 0.9f) return false;
    
    return true;
}

    float3 temporalReuse(const uint2 pixel , const bool validUV , const uint2 prevPixel , GeneralObjectInfo info , const float3 L)
    {
        if(validUV)
        if(checkDisocclusion(pixel , prevPixel))
        {
            float rcHitDistance;
            const float3 prevP = prevPositionImage[prevPixel].xyz;
            const float3 prevColor = RC::processRC(L , prevP , rcHitDistance, false);
            return prevColor;
        }

        return 0.0f;
    }
}

void processPixel(inout RayPayLoad load , const GeneralObjectInfo info , const uint2 pixel){
    uint2 size = DispatchRaysDimensions().xy;
    float2 uv = (float2(pixel) + 0.5) / float2(size);
    float4 clipPos = mul(generalBuffer.prevViewProj , float4(info.P , 1.0f));
    float3 pos = clipPos.xyz / clipPos.w;
    float2 prevUV = pos.xy * 0.5f + 0.5f;
    prevUV.y = 1.0f - prevUV.y;
    const bool validUV = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    // the pixel of the last frame that the point falls into . it used to be uint2( prevUV * size - 0.5 ) : it cuts to the pixel on the left / top of the nearest ones , so the
    // history was read from a place that is up to a pixel off in ONE direction every frame , and the accumulated shadow drifted that way ( it floated )
    uint2 prevPixel = min(uint2(max(prevUV * float2(size) , 0.0f)) , size - 1u);
    uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
    findLightIndices(info.P , lightIndices , lightCount);

    ReservoirDI di = ReSTIR_DI_generation(pixel ,prevUV , load.seed , info , lightIndices , lightCount );

    float3 totalDiffuse = 0.0f;
    float3 totalSpecular = 0.0f;   
    float totalShadow = 0.0f;
    float3 totalIndirect = 0.0f;
    float sunShadow = 1.0f;
    float lightShadow = 1.0f;
    float3 nextDir = 0.0f;
    float sunLuminance = 0.0f;

    const bool needSpecular = (info.roughness < 0.5f || info.metallic >= 0.7f);
    const bool needDiffuse  = ((1.0f - info.metallic) > 0.3f);

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
    const float3 rayOrigin = offsetRay(info.P , info.N , generalBuffer.playerPosition);

    float diffuseHitDistance = 0.0f;
    float specularHitDistance = 0.0f;

    // the ao map of the material takes the DIRECT light only ( the sun and the lamps ) : not the indirect light , not the ambient one and not the reflection ray
    // the strength is the same slider as the one of the SSAO / HBAO ( features.AO_intens ) : the same pow as in PostProcess
    const float objectAmbient = pow(objectAmbientImage[pixel].x , features.AO_intens);
    float3 reflectedSpecular = 0.0f;

    if(needSpecular && !load.fromReflection){
        float3 L; float pdf;
        ReSTIRGI::sampleSpecularDirection(float2(randNoise() , randNoise()) , info.N , info.V , info.roughness , L , pdf);
        if(pdf > 0.0f){
            const float3 fcos = ReSTIRGI::evalBRDF(info.N , info.V , L , info.roughness , F0);
            float distance;
            const float3 radiance = traceRayReflection(L , rayOrigin , distance);
            const float3 reflectedColor =  radiance * fcos / pdf;
            nextDir = L;
            reflectedSpecular += reflectedColor;
            specularHitDistance = distance;
        }          
    }

    ReservoirGI gi = (ReservoirGI)0;
    if(needDiffuse)
        {
        if(features.ReSTIRGI)
        {
            if(load.fromReflection){
                gi = ReSTIRGI::generateReservoirGI(info , pixel , load.seed);
            }else{
                gi = ReSTIRGI::ReSTIR_GI_generation(load.seed , info);
            }
            if(gi.M > 0u){
                const float3 L = normalize(gi.samplePos - info.P);
                diffuseHitDistance = length(gi.samplePos - info.P);
                totalIndirect = gi.radiance * ReSTIRGI::evalDiffuse(info.albedo , info.N , L , info.metallic) * gi.W;
            }
        }
        else if(features.RadianceCascade)
        {
            // the distance of the hit that the probes see : it is not used yet ( diffuseHitDistance is for the radius of the filter of the denoiser )
            float rcHitDistance = 0.0f;
            // we need halfVector
            const bool reflective = ReSTIRGI::isSpecularMode(info.roughness ,info.metallic);
            if(!reflective)
            {
                float pdf= 0.0f;
                float3 L = 0.0f;
                ReSTIRGI::sampleDiffuseDirection(float2(randNoise() , randNoise()) , info.N , L , pdf);
                float3 rcResult = 0.0f;
                if(pdf > 0.0f)
                {
                    const float3 brdf = ReSTIRGI::evalDiffuse(info.albedo , info.N , L , info.metallic);
                    rcResult  = brdf * RC::processRC(L , info.P , rcHitDistance, true) / pdf;
                    diffuseHitDistance = rcHitDistance;
                }   

                if(validUV)
                {
                    const float3 prevRC = RC::temporalReuse(pixel , validUV , prevPixel , info, L);
                    const float currentLum = luminance(rcResult);
                    const float prevLum = luminance(prevRC);
                    // the divider has a floor : two dark values ( 0 / 0 ) would be NaN , and NaN goes through the lerp into the picture and the denoiser
                    const float diff = abs(prevLum - currentLum) / max(max(currentLum , prevLum) , 1e-4f);   
                    const float k = smoothstep(0.0f , RC::MAX_RADIANCE_TOLERANCE , diff);
                    const float alpha = lerp(0.7f , 1.0f , k);
                    rcResult = lerp(prevRC , rcResult , alpha);
                }

                totalIndirect = rcResult;
            }
            else
            {
                float pdf= 0.0f;
                float3 L = 0.0f;
                float3 rcResult = 0.0f;

                ReSTIRGI::sampleSpecularDirection(float2(randNoise() , randNoise()) , info.N , info.V , info.roughness , L , pdf);
                if(pdf > 0.0f)
                {
                    const float3 fcos = ReSTIRGI::evalBRDF(info.N , info.V , L , info.roughness , F0);
                    rcResult = fcos * RC::processRC(L , info.P , rcHitDistance, true) / pdf;
                    diffuseHitDistance = rcHitDistance;
                }   
        
                if(validUV)
                {
                    const float3 prevRC = RC::temporalReuse(pixel , validUV , prevPixel , info, L);
                    const float currentLum = luminance(rcResult);
                    const float prevLum = luminance(prevRC);
                    // the divider has a floor : two dark values ( 0 / 0 ) would be NaN , and NaN goes through the lerp into the picture and the denoiser
                    const float diff = abs(prevLum - currentLum) / max(max(currentLum , prevLum) , 1e-4f);   
                    const float k = smoothstep(0.0f , RC::MAX_RADIANCE_TOLERANCE , diff);
                    const float alpha = lerp(0.7f , 1.0f , k);
                    rcResult = lerp(prevRC , rcResult , alpha);
                }

                totalIndirect = rcResult;
            }

        }
    }
    // no weights ( directWeight , indirectWeight ) for now : the clean colour
    
    // the luminance of the indirect light is NOT used in the shadow : the formulas of the sun and of the lamps get 0 ( the real value is the blocky indirect light of the
    // probes , it put the squares of the tiles into the shadow , and its history did not clear )
    const float indirectLum = 0.0f;

    if(sunBuffer.intensity > 0.0f){
        // sun
        // the direction of the sun is the direction of the LIGHT : the vector to the sun is the opposite one , and it has to be a unit vector
        const float3 L = randomInSunRadius(load.seed , sunBuffer , -sunBuffer.direction);
        const float sunNdotL = saturate(dot(info.N , L));
        const float3 diff = sunBuffer.diffuse;
        float3 diffuse = 0.0f;
        float3 specular = 0.0f;
        sunLuminance = luminance(sunBuffer.diffuse) * smoothstep(0.0f, 0.7f , dot(L , info.N));
        computeLight(info , L , diffuse , specular);
        // a surface that looks away from the sun is not lit by it whatever is in the way : no rays ( the weight of the sun is 0 there too )
        if(features.RayQuerySunSampleCount > 0 && sunNdotL > 0.0f){
            sunShadow =0.0f;
            const float sunCosAlpha = cos(sunBuffer.sourceRadius);
            const float3 sunOrigin = pos + L * 0.01f;
            [loop]
            for(uint i = 0 ; i < features.RayQuerySunSampleCount ; i++){
                const float3 dir = randomInSunRadius(load.seed , sunBuffer , -sunBuffer.direction);
                float rawShadow = calculateSunShadow(offsetRay(info.P , info.N , generalBuffer.playerPosition) , dir , 1000.0f , features.sunShadowFade);
                rawShadow = (sunLuminance * rawShadow + indirectLum) / max(indirectLum + sunLuminance , 1e-20);
                sunShadow += rawShadow;
            }

            sunShadow /= float(features.RayQuerySunSampleCount);
        }

        specular = lerp(0.0f, specular , sunShadow);

        totalDiffuse += diffuse * diff * sunNdotL;
        totalSpecular += specular * diff * sunNdotL;
    }

    float lightLuminance = 0.0f;
    // light here should be chosen by the forward + , not world grid light poses
    uint tileLightCount , tileLightOffset;
    findLightIndices(pixel , tileLightOffset , tileLightCount);
    {
        // lights
        if(features.RayQueryLightSampleCount > 0 && tileLightCount > 0){
            float weightSum = 0.0f;
            float visibleSum = 0.0f;
            [loop]
            for(uint i = 0 ; i < tileLightCount ;i++){
                const uint lightID = tileLightIndices[i + tileLightOffset];
                lg::LightSource light = lightBuffer[lightID];
                float3 L = light.position - info.P;
                L = normalize(L);
                const float3 lightPos = randomInLightRadius(load.seed , light , L);
                float3 dir = lightPos - info.P;
                const float d = length(dir);
                dir = normalize(dir);
                float shadow = 0.0f;
                const float reach = lightShadowReach(d , light.radius , features.lightShadowFade);
                const float falloff = 1.0f / (1.0f + light.Linear * d + light.Quadratic * d * d);
                const float weight = luminance(light.diffuse) * falloff * saturate(dot(info.N , dir)) * reach;
                if(reach <= 0.0f || weight <= 0.0f) continue;
                const float sinAlpha = min(light.radius / max(d , light.radius) , MAX_LIGHT_CONE_SIN);
                const float cosAlpha = sqrt(1.0f - sinAlpha * sinAlpha);
                const float att = lightAttenuation(light , d);
                const float currentLightLum = luminance(light.diffuse * att * reach) * saturate(dot(info.N , L));

                [loop]
                for(uint j = 0 ; j < features.RayQueryLightSampleCount ;j++){
                    float currentShadow = calculateShadowInSphere(load.seed , light , info.P , info.N);
                    currentShadow = (currentLightLum * currentShadow + indirectLum) / max(indirectLum + currentLightLum , 1e-20);
                    shadow += clamp(currentShadow , 0.0f , 1.0f);
                }
                shadow /= float(features.RayQueryLightSampleCount);
                shadow = lerp(1.0f , shadow , reach);
                weightSum += weight;
                visibleSum += weight * shadow;
                lightLuminance += currentLightLum;
            }
            if(weightSum > 0.0f) lightShadow = visibleSum / weightSum;

        }
        di.specularColor = lerp(0.0f, di.specularColor , lightShadow);
        totalDiffuse += di.diffuseColor.xyz;
        totalSpecular += di.specularColor.xyz;
    }


    const float sunShadowLifted = lerp(features.shadowMinimum , 1.0f , sunShadow);
    const float totalWeight = sunLuminance + lightLuminance;
    totalShadow = (totalWeight > 1e-6f) ? (sunLuminance * sunShadowLifted + lightLuminance * lightShadow) / totalWeight : 1.0f;
    uint shadowFrames = 1u;
    // the history counts only if it is the same surface : the position and the normal of the last frame ( checkDisocclusion ) , not only the value of the shadow
    if(validUV && RC::checkDisocclusion(pixel , prevPixel)){
        const float2 prevShadowData = prevShadowImage[prevPixel].xy;
        uint count = uint(prevShadowData.y);
        const float prevShadow = prevShadowData.x;
        const float diff = abs(prevShadow - totalShadow);
        const float k = smoothstep(0.0f , SHADOW_HISTORY_TOLERANCE , diff);
        count = min(count + 1u , MAX_SHADOW_FRAME);
        const float baseAlpha = 1.0f / float(count);
        const float alpha = lerp(baseAlpha , 1.0f , k);
        totalShadow = lerp(prevShadow , totalShadow , alpha);
        if(alpha < 0.8f)
        {
            shadowFrames = count;
        }
    }

    const float3 albedoSafe = max(info.albedo , 0.02f);
    const float3 diffuseOut = (totalDiffuse * objectAmbient) / albedoSafe;
    load.pixelInfo.albedo = info.albedo;

    load.lightInfo.diffuseHitDistance = min(diffuseHitDistance , 10000.0f);
    if(!load.fromReflection) load.lightInfo.specularHitDistance = min(specularHitDistance , 10000.0f);
    // the direction to the camera : ReLAX_temporal reflects it by the normal to find the virtual position ( it was a POINT here , reflect() of a point is garbage )
    load.pixelInfo.viewDir = info.V;
    load.lightInfo.shadow = totalShadow;
    load.lightInfo.shadowFrames = float(shadowFrames);
    load.lightInfo.diffuse = diffuseOut;
    load.lightInfo.indirect = totalIndirect / albedoSafe;
    load.lightInfo.specular = totalSpecular * objectAmbient + reflectedSpecular;
    load.pixelInfo.motionVector = prevUV - uv;
    if(!load.fromReflection) ReSTIRGIBuffer[bufferIndex(pixel)] = gi;
    if(!load.fromReflection) ReSTIRDIBuffer[bufferIndex(pixel)] = di;
}


// =============================== the hit of a REFLECTION ray ===============================
// a ray of the camera hits the surface that the G buffer holds in this PIXEL : processPixel reads the G buffer . but the ray of a reflection hits another surface , the one
// that the mirror shows , and the G buffer of the pixel of the mirror is the mirror itself : the reflected surface was lit with the position , the normal and the albedo of the
// mirror ( it came out black or grey ) . so the hit of a reflection takes everything from the hit : the material from the LUT and the textures ( readHitData ) , the lamps from
// the world grid by the position of the HIT ( computeLightAt ) , and it has no pixel of its own ( no reservoirs , no history )
namespace Reflection{

    struct HitSurface{
        float3 normal;
        float3 tangent;
        float3 bitangent;
        float2 uv;
    };

    // the attributes of the triangle in the object space , interpolated by the barycentrics . the indices of a mesh are LOCAL ( the draw has vertexOffset = firstVertex ,
    // the BLAS has firstVertex ) , so the vertex is firstVertex + index
    HitSurface interpolateHit(const uint meshID , const uint primitive , const float2 barycentrics){
        const MeshData mesh = meshBuffer[meshID];
        const uint base = mesh.firstIndex + primitive * 3u;

        const uint i0 = mesh.firstVertex + indices[base + 0u];
        const uint i1 = mesh.firstVertex + indices[base + 1u];
        const uint i2 = mesh.firstVertex + indices[base + 2u];

        const float w0 = 1.0f - barycentrics.x - barycentrics.y;
        const float w1 = barycentrics.x;
        const float w2 = barycentrics.y;

        HitSurface surface;
        surface.normal    = vertices[i0].normal    * w0 + vertices[i1].normal    * w1 + vertices[i2].normal    * w2;
        surface.tangent   = vertices[i0].tangent   * w0 + vertices[i1].tangent   * w1 + vertices[i2].tangent   * w2;
        surface.bitangent = vertices[i0].bitangent * w0 + vertices[i1].bitangent * w1 + vertices[i2].bitangent * w2;
        surface.uv        = vertices[i0].UV        * w0 + vertices[i1].UV        * w1 + vertices[i2].UV        * w2;
        return surface;
    }

    // the same as the one of the fragment shader , but SampleLevel : a ray has no derivatives
    float2 calculateParallaxAtHit(const float2 currentUV , const float3 V , const int depthID){
        if(depthID == NON_VALID_SLOT) return currentUV;
        const uint physicalIndex = virtualTextureIndex[depthID];
        const float depth = allTextures[physicalIndex].SampleLevel(typicalSampler , currentUV , 0).x;
        const float height = 1.0f - depth;
        const float height_scale = 0.001f;
        const float2 p = (V.xy) / V.z * height * height_scale;
        return currentUV - float2(p.x , -p.y);
    }

    // the surface that the ray hit : the position , the normal ( with the normal map ) , the albedo , the roughness and the metallic from the textures of its LUT
    GeneralObjectInfo readHitData(const float2 barycentrics){
        const InstanceData instance = instanceLayout[InstanceID()];
        const uint meshID = instance.meshDataIndex;
        const LUT lut = LUT_Buffer[instance.LUT_index];

        const HitSurface surface = interpolateHit(meshID , PrimitiveIndex() , barycentrics);

        // the same model matrix as the vertex shader of the Deferred render uses
        const float4x4 model = instanceBuffer[meshBuffer[meshID].instanceIndex];

        // normalMatrix3x3 already is inverse( transpose( model ) ) ( see the vertex shader ) , the matrix is on the left
        const float3 N = normalize(mul(normalMatrix3x3((float3x3)model) , surface.normal));
        const float3 T = normalize(mul(model , float4(surface.tangent , 0.0f)).xyz);
        const float3 B = normalize(mul(model , float4(surface.bitangent , 0.0f)).xyz);
        const float3x3 TBN = float3x3(T , B , N);

        const float3 hitPoint = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        // the direction back to the point that the ray comes from
        const float3 V = -WorldRayDirection();
        const float3 V_TBN = mul(TBN , V);

        const float2 uv = calculateParallaxAtHit(surface.uv , V_TBN , lut.depthID);

        float3 normal = N;
        if(lut.normalID != NON_VALID_SLOT){
            const uint nID = virtualTextureIndex[lut.normalID];
            float3 n;
            n.xy = allTextures[nID].SampleLevel(typicalSampler , uv , 0).xy * 2.0f - 1.0f;
            n.z = sqrt(saturate(1.0f - dot(n.xy , n.xy)));
            // OpenGL maps ( green up ) are used as they are with this TBN , the DirectX ones ( green down ) are flipped
            n.y = (generalBuffer.normalMapOpenGL != 0u) ? n.y : -n.y;
            normal = normalize(mul(n , TBN));
        }

        float3 albedo = lut.color;
        if(lut.diffuseID != NON_VALID_SLOT){
            const uint id = virtualTextureIndex[lut.diffuseID];
            albedo = allTextures[id].SampleLevel(typicalSampler , uv , 0).xyz;
        }

        float metallic = lut.metallic;
        if(lut.metallicTextureID != NON_VALID_SLOT){
            const uint id = virtualTextureIndex[lut.metallicTextureID];
            // RGBA8 map : a grey picture has the value in every channel , an ARM map has the metallic in B
            metallic = allTextures[id].SampleLevel(typicalSampler , uv , 0).z;
        }

        float roughness = lut.roughness;
        if(lut.roughnessTextureID != NON_VALID_SLOT){
            const uint id = virtualTextureIndex[lut.roughnessTextureID];
            // RGBA8 map : a grey picture has the value in every channel , an ARM map has the roughness in G
            roughness = allTextures[id].SampleLevel(typicalSampler , uv , 0).y;
        }

        // the packed PBR map ( R occlusion , G roughness , B metallic ) : the occlusion is not used yet
        if(lut.armTextureID != NON_VALID_SLOT){
            const uint id = virtualTextureIndex[lut.armTextureID];
            const float3 arm = allTextures[id].SampleLevel(typicalSampler , uv , 0).xyz;
            roughness = arm.y;
            metallic = arm.z;
        }

        // the map of the ambient occlusion : not used yet
        if(lut.aoTextureID != NON_VALID_SLOT){
        }

        GeneralObjectInfo info = (GeneralObjectInfo)0;
        info.albedo = albedo;
        info.metallic = metallic;
        // the GGX of a perfect mirror ( roughness 0 ) is degenerate : D is 0 almost everywhere , the surface came out black
        info.roughness = max(roughness , 0.1f);
        info.N = normal;
        info.V = V;
        info.P = hitPoint;
        return info;
    }

    // the light of the hit : the sun , one lamp of the world grid ( by the position of the HIT ) and the ambient of the sky ( kD ) , like the hits of the probes of
    // Radiance Cascade . the result is the radiance that goes back along the ray , the albedo is in it ( diffuse * albedo + specular )
    float3 computeLightAt(const GeneralObjectInfo data , inout uint seed , const float3 V){
        uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
        findLightIndices(data.P , lightIndices , lightCount);

        const float3 saveAlbedo = max(data.albedo , 1e-20f);

        // the diffuse part has no albedo while it is summed ( it is multiplied at the end ) , the specular part has none at all
        float3 diffuseSum = 0.0f;
        float3 specularSum = 0.0f;

        if(lightCount){
            const uint id = (uint)(rand(seed) * lightCount);
            lg::LightSource light = lightBuffer[lightIndices[id]];
            float3 diffuse , specular;

            float3 L = light.position - data.P;
            L = normalize(L);
            // the centre of the lamp , not a random point of its sphere ( the indirect light : a big sphere reached through the walls )
            const float3 lightPos = light.position;
            float3 realL = lightPos - data.P;
            const float dist = length(realL);
            realL = normalize(realL);

            computeLight(data , realL , diffuse , specular);

            const float NdotL = saturate(dot(data.N , realL));
            const float shadow = calculateShadowToCenter(light , data.P , data.N);
            const float window = lightAttenuation(light , dist);
            const float3 energy = NdotL * light.diffuse * shadow * window * float(lightCount);
            diffuseSum += (diffuse / saveAlbedo) * energy;
            specularSum += specular * energy;
        }

        if(sunBuffer.intensity > 0.0f)
        {
            float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
            const float3 L = randomInSunRadius(seed , sunBuffer , -sunBuffer.direction);
            const float sunShadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , 1000.0f);
            computeLight(data , L , sunDiffuse , sunSpecular);
            const float energy = saturate(dot(data.N , L)) * sunShadow;
            diffuseSum += (sunDiffuse / saveAlbedo) * sunBuffer.diffuse * energy;
            specularSum += sunSpecular * sunBuffer.diffuse * energy;
        }

        {
            const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , data.albedo , data.metallic);
            const float NdotV = max(dot(data.N , V) , 0.0001f);
            const float3 F = F_fresnel(F0 , NdotV);
            const float3 kD = (1.0f - F) * (1.0f - data.metallic);
            diffuseSum += sampleSkybox(data.N) * kD;
        }

        return diffuseSum + specularSum;
    }
}

// the payload that traceRayReflection expects : lightInfo.diffuse * max( albedo , 0.02 ) is the light , the others are 0 , the shadow is 1 ( the shadow is in the light already )
void processReflectionHit(inout RayPayLoad load , const GeneralObjectInfo info){
    const float3 light = Reflection::computeLightAt(info , load.seed , info.V);
    const float3 albedoSafe = max(info.albedo , 0.02f);
    const float hitDistance = RayTCurrent();

    load.lightInfo.diffuse = light;
    load.lightInfo.specular = float3(0.0f , 0.0f , 0.0f);
    load.lightInfo.indirect = float3(0.0f , 0.0f , 0.0f);
    load.lightInfo.shadow = 1.0f;
    load.lightInfo.shadowFrames = 0.0f;
    load.lightInfo.diffuseHitDistance = min(hitDistance , 10000.0f);
    load.lightInfo.specularHitDistance = min(hitDistance , 10000.0f);

    // every field of the payload is written : the caller reads all of them
    load.pixelInfo.position = info.P;
    load.pixelInfo.normal = info.N;
    load.pixelInfo.albedo = info.albedo;
    load.pixelInfo.viewDir = info.V;
    load.pixelInfo.motionVector = float2(0.0f , 0.0f);
    load.pixelInfo.depth = hitDistance;
    load.pixelInfo.metallic = info.metallic;
    load.pixelInfo.roughness = info.roughness;
    load.pixelInfo.pixelType = COLOR_PIXEL;
}

[shader("closesthit")]
void main(inout RayPayLoad load , in BuiltInTriangleIntersectionAttributes attr){
    // the state of randNoise is private to the entry point : without this every pixel and every frame got the same numbers
    initNoise(DispatchRaysIndex().xy , generalBuffer.frameIndex);
    load.depth++;
    if(load.depth >= MAX_RAY_CAST) return;
    // the ray of a reflection hits another surface than the one of the G buffer of this pixel : it reads the hit itself
    if(load.fromReflection){
        processReflectionHit(load , Reflection::readHitData(attr.barycentrics));
        return;
    }

    uint2 pixel = DispatchRaysIndex().xy;
    GeneralObjectInfo info = infoFromGBuffer(pixel);
    processPixel(load , info , pixel);
}