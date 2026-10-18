#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RaytracingAccelerationStructure TLAS;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(3 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(4 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(5 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(6 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(7 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(8 , 0)]] RWStructuredBuffer<LightSource> lightBuffer;

[[vk::binding(9 , 0)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(10 , 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(11 , 0)]] RWStructuredBuffer<InstanceData> instanceLayout;
[[vk::binding(12 , 0)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(14 , 0)]] RWStructuredBuffer<LUT> LUT_Buffer;

[[vk::binding(15 , 0)]] RWStructuredBuffer<ReservoirDI> ReSTIRDIBuffer;
[[vk::binding(16 , 0)]] RWStructuredBuffer<ReservoirDI> prevReSTIRDIBuffer;
[[vk::binding(17 , 0)]] RWStructuredBuffer<ReservoirGI> ReSTIRGIBuffer;
[[vk::binding(18, 0)]] RWStructuredBuffer<ReservoirGI> prevReSTIRGIBuffer;

[[vk::binding(19, 0)]] TextureCube skybox;
[[vk::binding(20, 0)]] SamplerState samp;
[[vk::binding(21 , 0)]] Texture2D transmissionLUT;
[[vk::binding(22, 0)]] Texture1D multiScatter;

[[vk::image_format("rgba16f")]]
[[vk::binding(23, 0)]] RWTexture2D<float4> diffuseImage;

[[vk::image_format("rgba16")]]
[[vk::binding(24, 0)]] RWTexture2D<float4> specularImage;

[[vk::image_format("rgba16")]]
[[vk::binding(25, 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("rgba16")]]
[[vk::binding(26, 0)]] RWTexture2D<float4> indirectImage;

[[vk::image_format("rgba16")]]
[[vk::binding(27, 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(28, 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("r32f")]]
[[vk::binding(29, 0)]] RWTexture2D<float> roughnessImage;

[[vk::image_format("r32f")]]
[[vk::binding(30, 0)]] RWTexture2D<float> metallicImage;

[[vk::image_format("r32f")]]
[[vk::binding(31, 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("r32f")]]
[[vk::binding(32, 0)]] RWTexture2D<float> shadowImage;

[[vk::image_format("r32f")]]
[[vk::binding(33, 0)]] RWTexture2D<float> prevShadowImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(34, 0)]] RWTexture2D<float4> viewImage;

[[vk::image_format("rg32f")]]
[[vk::binding(35, 0)]] RWTexture2D<float2> motionVector;

[[vk::image_format("r32f")]]
[[vk::binding(36, 0)]] RWTexture2D<float> pixelCheck;



struct LightData{
    float3 diffuse;
    float3 specular;
    float3 indirect;
    float shadow;
    // the REAL distances ( not weights ) : the denoisers take the radius of the filter from them . diffuse and indirect share the first one
    float diffuseHitDistance;
    float specularHitDistance;
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

void findLightIndices(uint2 currentPixel , out uint firstLight , out uint lightCount){
    firstLight = 0;
    lightCount = 0;

    uint2 tileCoord = currentPixel / forwardInfo.TILE_SIZE;
    uint tileIndex = tileCoord.x + tileCoord.y * forwardInfo.TILE_PER_ROW;

    // forward.hlsl's InterlockedAdd keeps counting past MAX_LIGHTS (it only stops WRITING there)
    lightCount = min(lightCountPerTile[tileIndex] , (uint)MAX_LIGHTS);
    firstLight = tileIndex * MAX_LIGHTS;
}

// the attenuation of a point light : the physical falloff times a window that reaches exactly 0 at lightRadius .
// without the window a light shone on EVERY pixel of the tiles it was listed in , far beyond its radius : the lit and the dark parts met
// on the edges of the Forward+ tiles ( hard vertical and horizontal borders ) and on the sphere of the radius
float lightAttenuation(LightSource light , float lightDistance){
    const float falloff = 1.0f / (1.0f + light.lightLinear * lightDistance + light.lightQuadratic * lightDistance * lightDistance);
    const float ratio = saturate(lightDistance / max(light.lightRadius , 1e-4f));
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
    const LightSource candidate = lightBuffer[lightIndex];
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

void finalizeReservoir_DI(inout ReservoirDI r , GeneralObjectInfo info){
    // an undefined light has no p_hat : lightBuffer must not be read with this index
    if(r.chosenIndex == 9999u){
        r.W = 0.0f;
        return;
    }
    const float p_hat = targetPdf_DI(r.chosenIndex , info);
    r.W = (p_hat > 1e-6f && r.M > 0u) ? r.weightSum / (float(r.M) * p_hat) : 0.0f;
}

void generateReservoir_DI(inout ReservoirDI res , inout uint seed , GeneralObjectInfo info, uint lightIndicesCount , uint lightOffset){
    res.W = 0.0f;
    res.M = 0;
    res.weightSum = 0.0f;
    res.chosenIndex = 9999u;   // undefined : no light is chosen
    res.diffuseColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
    res.specularColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
    if(lightIndicesCount == 0u) return;

    float bestPdf = 0.0f;
    for(uint i = 0 ; i < lightIndicesCount ; i++){
        const uint lightIndex = tileLightIndices[lightOffset + i];
        const LightSource light = lightBuffer[lightIndex];

        float3 toLight = light.position - info.P;
        const float lightDistance = length(toLight);
        toLight /= max(lightDistance , 1e-4f);

        float3 candidateDiffuse;
        float3 candidateSpecular;
        computeLight(info , toLight , candidateDiffuse , candidateSpecular);

        const float3 radiance = light.diffuse * lightAttenuation(light , lightDistance) * max(dot(info.N , toLight) , 0.0001f);
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

ReservoirDI ReSTIR_DI_generation(uint2 currentPixel , float2 prevUV , inout uint seed , GeneralObjectInfo info,  uint lightIndicesCount , uint lightOffset){
    const bool validUV = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    const uint2 prevCoords = (uint2)clamp(prevUV * float2(generalBuffer.screenWidth , generalBuffer.screenHeight) ,
                                          float2(0.0f , 0.0f) ,
                                          float2(generalBuffer.screenWidth - 1u , generalBuffer.screenHeight - 1u));

    // every light is visited in generateReservoir_DI , so there is nothing to collect from the last frame or from the neighbours . ( the reuse picks between
    // the reservoirs by a random number : it would bring the flashes back )
    ReservoirDI r;
    generateReservoir_DI(r , seed , info , lightIndicesCount , lightOffset);
    const uint M_CAP = 500u;
    if(r.M > M_CAP){
        r.weightSum *= float(M_CAP) / float(r.M);
        r.M = M_CAP;
    }
    return r;
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
    TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_NON_OPAQUE , 0xFF , RT_HITGROUP_SHADOW , 0 , RT_MISS_SHADOW , ray, shadow);
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
    TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_NON_OPAQUE , 0xFF , RT_HITGROUP_SHADOW , 0 , RT_MISS_SHADOW , ray, shadow);
    float k = 1.0f / shadow.depth;
    k = 1.0f - k;
    // the far object is , the lower shadow affect it , will look more like an ambient tbn
    const float data = (shadow.inShadow )? clamp(k , 0.0f , 1.0f) : 1.0f;
    return data;
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
        TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_NON_OPAQUE , 0xFF , RT_HITGROUP_SHADOW , 0 , RT_MISS_SHADOW , ray, shadow);
        return shadow.inShadow ? 0.0f : 1.0f;
    }

    // no ACCEPT_FIRST_HIT : the closest hit shader runs for the NEAREST accepted hit ( the lamp bodies are ignored by the any hit ) , and its depth is what is needed
    TraceRay(TLAS, RAY_FLAG_FORCE_NON_OPAQUE , 0xFF , RT_HITGROUP_SHADOW , 0 , RT_MISS_SHADOW , ray, shadow);
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

float3 randomInLightRadius(inout uint seed  , LightSource light , const float3 L){
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

float calculateShadowInSphere(inout uint seed , LightSource light , const float3 P , const float3 N){
    float3 L = light.position - P;
    const float d = length(L);
    L = normalize(L);
    const float3 randomPos = randomInLightRadius(seed , light , L);
    // the ray ends at the chosen point of the sphere , not at its centre ( offsetRay wants the position of the camera as the third argument , not a distance )
    const float3 toSample = randomPos - P;
    const float sampleDistance = length(toSample);
    const float3 dir = toSample / max(sampleDistance , 1e-4f);
    return calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , dir , sampleDistance);
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
        const float4 clipPos = mul(generalBuffer.viewProj, float4(info.worldPos , 1.0f));
        float2 prevUV = (clipPos.xy / max(clipPos.w , 1e-20)) * 0.5f + 0.5f;
        prevUV.y = 1.0f - prevUV.y;
        const bool validUV = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
        uint lightCount , lightOffset;
        uint2 pixel = DispatchRaysIndex().xy;

        if(validUV){
            const uint2 id = uint2(prevUV * float2(DispatchRaysDimensions().xy) - 0.5f);
            findLightIndices(id , lightOffset , lightCount);
        }else{
            findLightIndices(pixel , lightOffset , lightCount);
        }
        GeneralObjectInfo data = (GeneralObjectInfo)0;
        data.albedo = info.albedo;
        data.metallic = info.metallic;
        data.roughness = info.roughness;
        data.N = info.normal;
        data.P = info.worldPos;

        float3 result = 0.0f ;

        if(lightCount){
            const uint id = (uint)(rand(seed) * lightCount) + lightOffset;
            LightSource light = lightBuffer[tileLightIndices[id]];
            float3 diffuse, specular;

            const float3 V = normalize(x1Pos - info.worldPos);
            data.V = V;
            float3 L = light.position - data.P;
            const float dist = length(L);
            L = normalize(L);
            computeLight(data ,L , diffuse , specular );

            const float NdotL = saturate(dot(data.N , L));  
            float shadow = calculateShadowInSphere(seed , light , data.P , data.N);
            float window = lightAttenuation(light, dist);
            result = NdotL * light.diffuse * (diffuse) * shadow  * window * lightCount;
        }
        if(sunBuffer.intensity > 0.0f){
            float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
            // now we also have to check if the shadow actually hits it , since if not, there is no point at computing the light , since its also the calcualtion
            const float sunShadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , -sunBuffer.direction , 1000.0f);
            computeLight(data , -sunBuffer.direction , sunDiffuse , sunSpecular);
            //
            result += (sunDiffuse) *  clamp(dot(data.N , -sunBuffer.direction),  0.0f , 1.0f) * sunBuffer.diffuse  * sunShadow;
        }

        // the sky light of the point x2 itself : a point in the shadow of the sun is not black , the sky lights it ( without it every ray of GI that hit
        // something in the shade brought 0 : the black spots of the indirect light that moved with the azimuth of the sun ) . the same form as the sky
        // ambient of the pixel : the average of the sky over the hemisphere of x2 * kD * the albedo of x2 ( the direction of the view is x1 )
        {
            const float3 V2 = normalize(x1Pos - info.worldPos);
            const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
            const float NdotV = max(dot(info.normal , V2) , 0.0001f);
            const float3 F = F_fresnel(F0 , NdotV);
            const float3 kD = (1.0f - F) * (1.0f - info.metallic);
            result += sampleSkybox(info.normal) * kD * info.albedo;
        }
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

    ReservoirGI temporalReuse_GI( inout uint seed ,ReservoirGI current , uint2 prevCoord , GeneralObjectInfo info){
        ReservoirGI r = (ReservoirGI)0;

        ReservoirGI prevGI = prevReSTIRGIBuffer[bufferIndex(prevCoord)];
        if(current.roughness < 0.4 || prevGI.roughness < 0.4f){
            if(!(current.roughness < 0.4f && prevGI.roughness < 0.4f)) return current;
        }

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

    ReservoirGI ReSTIR_GI_generation(inout uint seed , GeneralObjectInfo info){
        ReservoirGI res = (ReservoirGI)0;
        if(!features.ReSTIRGI){
            res.W = 1.0f;
            res.radiance = 1.0f;
            return res;
        }

        uint2 pixel = DispatchRaysIndex().xy;

        res = generateReservoirGI(info , pixel  , seed);
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
                res = temporalReuse_GI(seed , res , prevPixel ,info);
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

float3 traceRayReflection(const float3 direction , const float3 position , out float hitDistance){

    RayPayLoad load = (RayPayLoad)0;
    load.fromReflection = true;

    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = 1000.0f;
    ray.TMin = 0.001f;

    TraceRay(TLAS , RAY_FLAG_NONE , 0xFF , RT_HITGROUP_GENERAL , 0 , RT_MISS_GENERAL , ray , load);

    hitDistance = load.lightInfo.specularHitDistance;
    if(load.pixelInfo.pixelType == SKYBOX_PIXEL){
        return load.lightInfo.diffuse;
    }

    // the diffuse of the hit is demodulated ( divided by max( albedo , 0.02 ) ) and holds the indirect light too : the albedo brings it back , the indirect
    // light is taken out of it and put back without the shadow ( the shadow belongs to the direct light only , the indirect light has its own visibility inside )
    const float3 indirect = load.lightInfo.indirect;
    const float3 directDiffuse = max(load.lightInfo.diffuse * max(load.pixelInfo.albedo , 0.02f) + indirect , 0.0f);
    return (directDiffuse + load.lightInfo.specular) * load.lightInfo.shadow + indirect* load.lightInfo.shadow;
}

void processPixel(inout RayPayLoad load , const GeneralObjectInfo info , const uint2 pixel){
    uint2 size = DispatchRaysDimensions().xy;
    float2 uv = (float2(pixel) + 0.5) / float2(size);
    float4 clipPos = mul(generalBuffer.prevViewProj , float4(info.P , 1.0f));
    float3 pos = clipPos.xyz / clipPos.w;
    float2 prevUV = pos.xy * 0.5f + 0.5f;
    prevUV.y = 1.0f - prevUV.y;
    const bool validUV = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    uint2 prevPixel = uint2(prevUV * float2(size) - 0.5f);
    uint lightCount , lightOffset;
    findLightIndices(pixel , lightOffset , lightCount);

    ReservoirDI di = ReSTIR_DI_generation(pixel ,prevUV , load.seed , info , lightCount , lightOffset );

    float3 totalDiffuse = 0.0f;
    float3 totalSpecular = 0.0f;   
    float totalShadow = 0.0f;
    float3 totalIndirect = 0.0f;
    float sunShadow = 1.0f;
    float lightShadow = 1.0f;
    float3 nextDir = 0.0f;
    float sunLuminance = 0.0f;

    if(sunBuffer.intensity > 0.0f){
        // sun
        // the direction of the sun is the direction of the LIGHT : the vector to the sun is the opposite one , and it has to be a unit vector
        const float3 L = normalize(-sunBuffer.direction);
        const float sunNdotL = saturate(dot(info.N , L));
        const float3 diff = sunBuffer.diffuse;
        float3 diffuse = 0.0f;
        float3 specular = 0.0f;
        computeLight(info , L , diffuse , specular);
        if(features.RayQuerySunSampleCount > 0 ){
            sunShadow =0.0f;
            const float3 toSun = normalize(-sunBuffer.direction);
            // the size of the sun for the shadow ( the angle of the cone , radians ) is sunBuffer.sourceRadius : bigger - a softer penumbra .
            // the disc in the sky still has SUN_ANGULAR_RADIUS
            const float sunCosAlpha = cos(sunBuffer.sourceRadius);
            const float3 sunOrigin = pos + toSun * 0.01f;
            [loop]
            for(uint i = 0 ; i < features.RayQuerySunSampleCount ; i++){
                const float3 dir = sampleCone(toSun , sunCosAlpha , load.seed);
                sunShadow += calculateSunShadow(offsetRay(info.P , info.N , generalBuffer.playerPosition) , dir , 1000.0f , features.sunShadowFade);
            }

            sunShadow /= float(features.RayQuerySunSampleCount);
        }
        totalDiffuse += diffuse * diff * sunNdotL;
        totalSpecular += specular * diff * sunNdotL;
        sunLuminance = luminance(sunBuffer.diffuse) * smoothstep(0.0f, 0.7f , dot(-sunBuffer.direction , info.N));
    }

    float lightLuminance = 0.0f;
    {
        // lights
        if(features.RayQueryLightSampleCount > 0 && lightCount > 0){
            // the weighted mean over the lamps of the tile ( the same as RayQueryShadows ) : a lamp that does not reach the pixel ( weight 0 , it is in the
            // list of the tile only because the test of the tile is conservative ) must not count at all . before it was skipped by continue , but the
            // sum was still divided by lightCount , so it counted as a full shadow : the ellipses of the radii and the rectangles of the tiles
            float weightSum = 0.0f;
            float visibleSum = 0.0f;
            [loop]
            for(uint i = 0 ; i < lightCount ;i++){
                const uint tileLightIndex = lightOffset + i;
                const uint lightID = tileLightIndices[tileLightIndex];
                LightSource light = lightBuffer[lightID];
                float3 L = light.position - info.P;
                const float d = length(L);
                L = normalize(L);
                float shadow = 0.0f;
                // the shadow of a lamp fades out beyond its radius ( lightShadowReach : 1 inside of the radius , 0 at lightRadius * ( 1 + lightShadowFade ) ) instead
                // of ending with a jump . the weight has no window of the radius any more : the window is in the reach
                const float reach = lightShadowReach(d , light.lightRadius , features.lightShadowFade);
                const float falloff = 1.0f / (1.0f + light.lightLinear * d + light.lightQuadratic * d * d);
                // the weight fades with the reach too : a lamp that ends its fade leaves the mean without a jump
                const float weight = luminance(light.diffuse) * falloff * saturate(dot(info.N , L)) * reach;
                if(reach <= 0.0f || weight <= 0.0f) continue;
                const float sinAlpha = min(light.lightRadius / max(d , light.lightRadius) , MAX_LIGHT_CONE_SIN);
                const float cosAlpha = sqrt(1.0f - sinAlpha * sinAlpha);
                [loop]
                for(uint j = 0 ; j < features.RayQueryLightSampleCount ;j++){
                    float currentShadow = calculateShadowInSphere(load.seed , light , info.P , info.N);
                    shadow += currentShadow;
                }
                shadow /= float(features.RayQueryLightSampleCount);
                // beyond the radius the shadow of this lamp is weaker and weaker : 1 ( no shadow ) at the end of the fade
                shadow = lerp(1.0f , shadow , reach);
                weightSum += weight;
                visibleSum += weight * shadow;
                // lightAttenuation : with the window of the radius , the same as the lighting itself ( LightPass , ReSTIR ) . without the window the lamp
                // had a weight at the edge of its radius where it does not light anything any more , and its shadow was seen there as an arc
                const float att = lightAttenuation(light , d);
                lightLuminance += luminance(light.diffuse * att) * saturate(dot(info.N , L));
            }
            if(weightSum > 0.0f) lightShadow = visibleSum / weightSum;
            // how much the lamps give to this pixel in total ( luminance * falloff * cos * reach of every lamp ) : the weight of lightShadow against sunShadow

        }

        totalDiffuse += di.diffuseColor.xyz;
        totalSpecular += di.specularColor.xyz;
    }


    // the mean of the two shadows by what each of the sources gives to the pixel . when none of them gives anything ( the sun is off , the lamps are not traced
    // or do not reach ) there is nothing to weigh : no shadow ( 1 ) . it used to be 0 / 1e-20 = 0 : the whole direct light of the lamps went out
    const float totalWeight = sunLuminance + lightLuminance;
    totalShadow = (totalWeight > 1e-6f) ? (sunLuminance * sunShadow + lightLuminance * lightShadow) / totalWeight : 1.0f;

    // temporal occlusion for shadow
    if(validUV){
        const float prevShadow = prevShadowImage[prevPixel].x;
        totalShadow = lerp(totalShadow , prevShadow , features.temporalShadowK);
    }
    // now the GI

    const bool needSpecular = (info.roughness < 0.5f || info.metallic >= 0.7f);
    const bool needDiffuse  = ((1.0f - info.metallic) > 0.3f);

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
    const float3 rayOrigin = offsetRay(info.P , info.N , generalBuffer.playerPosition);

    float diffuseHitDistance = 0.0f;
    float specularHitDistance = 0.0f;

    if(needSpecular && !load.fromReflection){
        float3 L; float pdf;
        ReSTIRGI::sampleSpecularDirection(float2(randNoise() , randNoise()) , info.N , info.V , info.roughness , L , pdf);

        if(pdf > 0.0f){
            const float3 fcos = ReSTIRGI::evalBRDF(info.N , info.V , L , info.roughness , F0);
            float distance;
            const float3 radiance = traceRayReflection(L , rayOrigin , distance);
            const float3 reflectedColor = radiance * fcos / pdf * info.metallic;
            nextDir = L;
            totalSpecular += reflectedColor;
            specularHitDistance = distance;
        }          
    }

    ReservoirGI gi = (ReservoirGI)0;
    if(load.fromReflection){
        gi = ReSTIRGI::generateReservoirGI(info , pixel , load.seed);
    }else{
        gi = ReSTIRGI::ReSTIR_GI_generation(load.seed , info);
    }
    if(gi.M > 0u){
        const float3 L = normalize(gi.samplePos - info.P);
        diffuseHitDistance = length(gi.samplePos - info.P);
        totalIndirect += gi.radiance * ReSTIRGI::evalDiffuse(info.albedo , info.N , L , info.metallic) * gi.W;
    }
    // no weights ( directWeight , indirectWeight ) for now : the clean colour

    const float3 albedoSafe = max(info.albedo , 0.02f);
    const float3 diffuseOut = (totalDiffuse) / albedoSafe;
    load.pixelInfo.albedo = info.albedo;

    load.lightInfo.diffuseHitDistance = min(diffuseHitDistance , 10000.0f);
    if(!load.fromReflection) load.lightInfo.specularHitDistance = min(specularHitDistance , 10000.0f);
    load.pixelInfo.viewDir = info.P + nextDir * specularHitDistance;
    load.lightInfo.shadow = totalShadow;
    load.lightInfo.diffuse = diffuseOut;
    load.lightInfo.indirect = totalIndirect / albedoSafe;
    load.lightInfo.specular = totalSpecular;
    load.pixelInfo.motionVector = prevUV - uv;
    if(!load.fromReflection) ReSTIRGIBuffer[bufferIndex(pixel)] = gi;
    if(!load.fromReflection) ReSTIRDIBuffer[bufferIndex(pixel)] = di;
}


[shader("closesthit")]
void main(inout RayPayLoad load , in BuiltInTriangleIntersectionAttributes attr){
    load.depth++;
    if(load.depth >= MAX_RAY_CAST) return;
    uint2 pixel = DispatchRaysIndex().xy;
    GeneralObjectInfo info = infoFromGBuffer(pixel);
    processPixel(load , info , pixel);
}