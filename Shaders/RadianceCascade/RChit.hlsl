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

[[vk::binding(15, 0)]] TextureCube skybox;
[[vk::binding(16, 0)]] SamplerState samp;
[[vk::binding(17 , 0)]] Texture2D transmissionLUT;
[[vk::binding(18, 0)]] Texture1D multiScatter;

[[vk::binding(0 , 1)]] StructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState textureSampler;
[[vk::binding(2 , 1)]] Texture2D allTextures[];

[[vk::binding(0 , 2)]] RWStructuredBuffer<RC::ProbeInfo> RCProbeBuffer;
[[vk::binding(1 , 2)]] RWStructuredBuffer<RC::CascadeInfo> RCCascadesBuffer;
[[vk::binding(2 , 2)]] ConstantBuffer<RC::GeneralProbeInfo> RCProbeInfo;
[[vk::image_format("rgba16f")]]
[[vk::binding(3 , 2)]] RWTexture2D<float4> RCImages[];

struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float depth : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

struct [raypayload] RayPayLoad{
    float3 finalColor : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint seed : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint cascadeID : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float alpha : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint mode : read(closesthit , miss, caller) : write(closesthit , miss , caller);
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
    TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_NON_OPAQUE , 0xFF , 1 , 0 , 1 , ray, shadow);
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
    TraceRay(TLAS, RAY_FLAG_FORCE_NON_OPAQUE , 0xFF , 1 , 0 , 1 , ray, shadow);
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
        TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_NON_OPAQUE , 0xFF , 1 , 0 , 1 , ray, shadow);
        return shadow.inShadow ? 0.0f : 1.0f;
    }

    // no ACCEPT_FIRST_HIT : the closest hit shader runs for the NEAREST accepted hit ( the lamp bodies are ignored by the any hit ) , and its depth is what is needed
    TraceRay(TLAS, RAY_FLAG_FORCE_NON_OPAQUE , 0xFF , 1 , 0 , 1 , ray, shadow);
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
    const float facing = dot(offset , L);
    if(facing > 0.0f) offset = -offset;
    return normalize(normalize(L) + offset);
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
    return calculateShadowAtt(offsetRay(P , N , generalBuffer.playerPosition) , dir , sampleDistance);
}

float3 stretchView(const float3 V , const float a){
    return normalize(float3(a  * V.x , V.y * a , V.z));
}

struct HitSurface{
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
};

// the attributes of the triangle in the object space , interpolated by the barycentrics . the indices of a mesh are LOCAL ( the draw has
// vertexOffset = firstVertex , the BLAS has firstVertex ) , so the vertex is firstVertex + index
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
    const float depth = allTextures[physicalIndex].SampleLevel(textureSampler , currentUV , 0).x;
    const float height = 1.0f - depth;
    const float height_scale = 0.001f;
    const float2 p = (V.xy) / V.z * height * height_scale;
    return currentUV - float2(p.x , -p.y);
}

GeneralObjectInfo readData(const float2 barycentrics){
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
        n.xy = allTextures[nID].SampleLevel(textureSampler , uv , 0).xy * 2.0f - 1.0f;
        n.z = sqrt(saturate(1.0f - dot(n.xy , n.xy)));
        // OpenGL maps ( green up ) are used as they are with this TBN , the DirectX ones ( green down ) are flipped . ( it was the other way round : every map was inverted )
        n.y = (generalBuffer.normalMapOpenGL != 0u) ? n.y : -n.y;
        normal = normalize(mul(n , TBN));
    }

    float3 albedo = lut.color;
    if(lut.diffuseID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.diffuseID];
        albedo = allTextures[id].SampleLevel(textureSampler , uv , 0).xyz;
    }

    float metallic = lut.metallic;
    if(lut.metallicTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.metallicTextureID];
        metallic = allTextures[id].SampleLevel(textureSampler , uv , 0).x;
    }

    float roughness = lut.roughness;
    if(lut.roughnessTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.roughnessTextureID];
        roughness = allTextures[id].SampleLevel(textureSampler , uv , 0).x;
    }

    GeneralObjectInfo info = (GeneralObjectInfo)0;
    info.albedo = albedo;
    info.metallic = metallic;
    info.roughness = roughness;
    info.N = normal;
    info.V = V;
    info.P = hitPoint;
    return info;
}

// the pixel of the screen where a point of the world is ( -999 behind the camera ) : the lamps of the Forward+ tile of that pixel light the point
int2 convertToScreenPixel(const float3 position){
    const float4 clipPos =mul(generalBuffer.viewProj , float4(position , 1.0f));
    const uint2 screenSize = int2(generalBuffer.screenWidth ,generalBuffer.screenHeight);
    if(clipPos.w <= 0.0f) return -999;
    float2 uv = (clipPos.xy / clipPos.w) * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;
    const int2 pixel = int2(uv * float2(screenSize ) - 0.5f);
    return pixel;
}

void processPixel(inout RayPayLoad load , in float2 barycentrics){
    GeneralObjectInfo info = readData(barycentrics);

    // the tile of the HIT point ( not of the probe : the probes are all around the player , most of them are off the screen )
    const int2 pixel = convertToScreenPixel(info.P);
    const int2 screenSize = int2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    const bool validPixel = all(pixel >= int2(0 , 0)) && all(pixel < screenSize);
    float3 totalDiffuse = 0.0f;
    float3 totalSpecular = 0.0f;

    // lets start with sun
    const float3 saveAlbedo = max(info.albedo  , 1e-20);
    float sunLuminance = 0.0f;

    {
        if(sunBuffer.intensity > 0.0f){
            float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
            // now we also have to check if the shadow actually hits it , since if not, there is no point at computing the light , since its also the calcualtion
            const float3 L = randomInSunRadius(load.seed , sunBuffer , -sunBuffer.direction);
            const float sunShadow = calculateShadows(offsetRay(info.P , info.N , generalBuffer.playerPosition) , L , 1000.0f);
            computeLight(info , L , sunDiffuse , sunSpecular);
            sunDiffuse /= saveAlbedo;
            totalDiffuse += (sunDiffuse) *  clamp(dot(info.N , L),  0.0f , 1.0f) * sunBuffer.diffuse  * sunShadow;
            totalSpecular = sunSpecular * sunShadow  *  clamp(dot(info.N , L),  0.0f , 1.0f) * sunBuffer.diffuse;
        }else{
            totalDiffuse +=  0.01f;
        }
    }
    float lightLuminance = 0.0f;
    
    // now for lights
    if(validPixel){
        uint lightCount , lightOffset;
        findLightIndices(pixel , lightOffset , lightCount);
        if(lightCount > 0){

            const uint id = (uint)(rand(load.seed) * lightCount) + lightOffset;
            LightSource light = lightBuffer[tileLightIndices[id]];
            float3 diffuse = 0.0, specular = 0.0f;
            float3 toLight = normalize(light.position - info.P);
            const float3 lightPos = randomInLightRadius(load.seed , light , toLight);
            float3 L = lightPos - info.P;
            const float d = length(L);
            L = normalize(L);
            if(calculateShadows(offsetRay(info.P , info.N , generalBuffer.playerPosition) , L , d) == 1.0f){
                computeLight(info , L , diffuse , specular);
                diffuse /= saveAlbedo;
                const float NdotL = saturate(dot(info.N , L));  
                float shadow = calculateShadowInSphere(load.seed , light , info.P , info.N);
                float window = lightAttenuation(light, d);
                totalDiffuse += NdotL * light.diffuse * (diffuse) * shadow  * window * lightCount;
                totalSpecular += NdotL * light.diffuse * (specular) * shadow  * window * lightCount;
            }else{
                totalDiffuse +=  0.01f;
            }

        }
    }

    load.finalColor = totalDiffuse * info.albedo + totalSpecular;
    // the ray hit something inside of its interval : what is behind the surface does not count
    load.alpha = 1.0f;
}

[shader("closesthit")]
void main(inout RayPayLoad load , in BuiltInTriangleIntersectionAttributes attr){
    // mode 1 : the ray that finds out if the probe is inside of an object ( RCrayGeneral.hlsl ) . from inside of a closed object the first hit is the BACK of a triangle
    // ( from the outside it is always a front face ) : the probe is in the geometry , its light is of no use ( the rays see the objects from inside ) and the gather and the merge
    // skip it . 255 is HIT_KIND_TRIANGLE_BACK_FACE . updatePoses writes validProbe = 1 for every probe at the start of the frame , the test only takes it away
    if(load.mode == 1u){
        if(HitKind() == 255u){
            RCProbeBuffer[RCCascadesBuffer[load.cascadeID].PROBE_OFFSET + DispatchRaysIndex().y].validProbe = 0u;
        }
        load.finalColor = 0.0f;
        load.alpha = 1.0f;
        return;
    }
    processPixel(load , attr.barycentrics);
}