#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RaytracingAccelerationStructure TLAS;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(3 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(4 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(5 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(6 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(7 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;

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

[[vk::binding(0 , 3)]] RWStructuredBuffer<lg::LightSource> lightBuffer;
[[vk::binding(1 , 3)]] RWStructuredBuffer<lg::LightCell> lightCells;
[[vk::binding(2 , 3)]] ConstantBuffer<lg::GridCascadeInfo> gridInfo;


static const float RC_PLANE_SIGMA_PER_PIXEL = 0.003f;
static const float RC_DIRECTION_POWER = 16.0f;
static const float RC_CONFIDENCE_FULL = 0.5f;
// the weight of a probe by the NORMAL ( dot of the normals ^ 16 ) never goes under this : in a corner every probe around the point lies on the OTHER wall ( dot = 0 ) ,
// the weight was 0 for all of them and the corner got black . the plane and the visibility checks stay , they are what keeps the light from the other side of a wall out
static const float RC_MIN_NORMAL_WEIGHT = 0.05f;
static const float MAX_RC_RADIANCE_LIMIT = 0.7f;

// the SPATIAL REUSE of the probes : 1 - the 2 x 2 probes and the 2 x 2 texels are mixed ( the bilinear ) with the weights of the geometry , of the direction and of the radiance .
// 0 - NO reuse at all : the nearest probe and the nearest texel of its map , the weight is 1 , nothing is checked ( for the search of what spoils the picture )
#define RC_SPATIAL_REUSE 1

// the weight of a texel of the map of directions by its distance ( in texels ) from the place that is looked for : the tent ( the bilinear ) or the SMOOTHED tent .
// smoothstep( 1 - d ) has the zero slope at the texel itself , and for the two neighbours of the window 2 x 2 the weights still add up to 1 ( smoothstep( t ) + smoothstep( 1 - t ) = 1 ) .
// 1 : smoothed , 0 : the plain tent
#define RC_SMOOTH_TEXEL_WEIGHT 1
float rcTexelWeight1D(const float distance , const float radius)
{
    const float tent = saturate(1.0f - distance / radius);
#if RC_SMOOTH_TEXEL_WEIGHT
    return smoothstep(0.0f , 1.0f , tent);
#else
    return tent;
#endif
}

struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float depth : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

struct [raypayload] RayPayLoad{
    float3 finalColor : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint seed : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    uint cascadeID : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float alpha : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float hitDistance : read(closesthit , miss, caller) : write(closesthit , miss , caller);
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
    TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH , 0xFF , 1 , 0 , 1 , ray, shadow);
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
    TraceRay(TLAS, RAY_FLAG_NONE , 0xFF , 1 , 0 , 1 , ray, shadow);
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
        TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH , 0xFF , 1 , 0 , 1 , ray, shadow);
        return shadow.inShadow ? 0.0f : 1.0f;
    }

    // no ACCEPT_FIRST_HIT : the closest hit shader runs for the NEAREST accepted hit ( the lamp bodies are ignored by the any hit ) , and its depth is what is needed
    TraceRay(TLAS, RAY_FLAG_NONE , 0xFF , 1 , 0 , 1 , ray, shadow);
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
    float3 N = normalize(mul(normalMatrix3x3((float3x3)model) , surface.normal));
    // a double sided surface ( the leaves , the cloth ) hit from the BACK : the interpolated normal looks away from the ray , and NdotV / NdotL of the BRDF are degenerate
    if(HitKind() == HIT_KIND_TRIANGLE_BACK_FACE) N = -N;
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
        // RGBA8 map : a grey picture has the value in every channel , an ARM map has the metallic in B
        metallic = allTextures[id].SampleLevel(textureSampler , uv , 0).z;
    }

    float roughness = lut.roughness;
    if(lut.roughnessTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.roughnessTextureID];
        // RGBA8 map : a grey picture has the value in every channel , an ARM map has the roughness in G
        roughness = allTextures[id].SampleLevel(textureSampler , uv , 0).y;
    }

    // the packed PBR map ( R occlusion , G roughness , B metallic ) : the same as the deferred render reads it , the occlusion is not used yet
    if(lut.armTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.armTextureID];
        const float3 arm = allTextures[id].SampleLevel(textureSampler , uv , 0).xyz;
        roughness = arm.y;
        metallic = arm.z;
    }


    // the map of the ambient occlusion : not used yet
    if(lut.aoTextureID != NON_VALID_SLOT){
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

float3 sampleSkybox(const float3 P ,const float3 N){
    float3 result = 0.0f;
    // 0 samples : no sky light ( and no division by 0 below )
    if(features.skyboxSampleCount == 0u) return result;
    [loop]
    for(uint i = 0 ; i  < features.skyboxSampleCount ; i++){
        const float3 direction = randomInHemisphere(N);
        const float shadow = calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , direction , 5.0f);
        result += skybox.SampleLevel(samp , direction , 0).xyz * shadow;
    }
    result /= float(features.skyboxSampleCount);
    result = lerp(0.0f , result , sunBuffer.intensity);
    return result;
}

namespace RC
{
    // the place of the world point P in the GRID OF THE PROBES of the cascade 0 , as a float : the probes are attached to the tiles of the SCREEN ( not to the world ) , so the
    // point is projected to the screen first . the probe i stands in the middle of its tile , so its centre is at the integer i in these units :
    //   floor( probePos )  - the probe on the left / top of the point , the next one is + 1 , and  probePos - floor( probePos )  is the weight of the next one ( the bilinear )
    // returns false if there is no probe for the point : it is behind the camera , or outside of the screen ( the probes of the screen know nothing about it )
    // it does not look at the depth : a point hidden behind something else gets the place of the pixel in front of it , the caller compares the depth / the normal if it matters
    bool findProbePosition(const float3 P , out float2 probePos , const bool current)
    {
        probePos = float2(0.0f , 0.0f);

        const float4 clipPos = mul((current ? generalBuffer.viewProj : generalBuffer.prevViewProj) , float4(P , 1.0f));
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
    int2 findProbe(const float3 P, const bool current)
    {
        float2 probePos;
        if(!findProbePosition(P , probePos , current)) return int2(-1 , -1);

        const RC::CascadeInfo cascade = RCCascadesBuffer[0];
        // the last row / column of the tiles is cut by the edge of the screen , but a probe is there : the number of the probes is what the image holds
        const int2 probeCount = int2(cascade.CASCADE_X , cascade.CASCADE_Y) / int(cascade.BUILD_SIZE);
        return clamp(int2(floor(probePos + 0.5f)) , int2(0 , 0) , probeCount - 1);
    }

    // hitDistance : the distance of the hit that the probes see in the direction ( meters ) , for the radius of the filter of the denoiser : the mean of the distances of the
    // hits ( the sky : 1000 ) with the same weights as the colour . it is 1000 when there is no probe
    float3 processRC(const float3 dir , const float3 P, const float3 N , out float hitDistance , const bool currentImage)
    {
        hitDistance = 1000.0f;
        float2 probePos;
        if(!findProbePosition(P , probePos, currentImage))
        {
            // the hit is off the screen ( the probes know nothing about it ) : there is no light to take from the cache . it is NOT the sky : the sky without the visibility lit the
            // interiors orange ( the sky light is in the cache already : the rays that left the room carry it )
            return 0.0f;
        }

        const RC::CascadeInfo c = RCCascadesBuffer[0];
        const int buildSize = int(c.BUILD_SIZE);
        const int2 probeCount = int2(c.CASCADE_X , c.CASCADE_Y) / buildSize;

#if RC_SPATIAL_REUSE

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
        // the weight of the valid neighbours without the geometry ( the normal , the plane , the visibility ) : see RC_CONFIDENCE_FULL
        float idealSum = 0.0f;

        // the REFERENCE of the limit of the radiance : the nearest texel of the nearest probe ( see MAX_RC_RADIANCE_LIMIT )
        float referenceLuminance = 0.0f;
        bool hasReference = false;
        {
            const int2 referenceProbe = clamp(int2(floor(probePos + 0.5f)) , int2(0 , 0) , probeCount - 1);
            const uint referenceIndex = c.PROBE_OFFSET + uint(referenceProbe.y) * c.PROBE_PER_ROW + uint(referenceProbe.x);
            if(RCProbeBuffer[referenceIndex].validProbe != 0u)
            {
                const int2 referenceTexel = clamp(int2(floor(texelPos + 0.5f)) , int2(0 , 0) , int2(buildSize - 1 , buildSize - 1));
                const uint2 referencePixel = uint2(referenceProbe * buildSize + referenceTexel);
                const float4 referenceData = (currentImage ? RCImages[RC::imageIndex(0u , generalBuffer.frameIndex)][referencePixel] : RCImages[RC::prevImageIndex(generalBuffer.frameIndex)][referencePixel]);
                referenceLuminance = luminance(referenceData.xyz);
                hasReference = true;
            }
        }

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
                const RC::ProbeInfo probe = RCProbeBuffer[probeIndex];
                const float2 probeDistance = abs(float2(probeX , probeY) - probeFrac);
                // the bilinear weight of the probe : the weight that it has when the geometry does not matter
                const float baseWeight = saturate(1.0f - probeDistance.x / float(PROBE_SAMPLE_COUNT)) * saturate(1.0f - probeDistance.y / float(PROBE_SAMPLE_COUNT));
                if(baseWeight <= 0.0f) continue;
                const float nW = max(pow(saturate(dot(N , probe.normal)) , 16.0f) , RC_MIN_NORMAL_WEIGHT);
                // the distance to the plane of the probe : the weight FALLS with it ( a probe of the neighbour wall is far from this plane ) . the scale is the size of a tile
                // of the probes in the world
                const float d = abs(dot(P - probe.position , probe.normal));
                const float planeSigma = max(probe.cameraDistance * float(c.PROBE_RESOLUTION) * RC_PLANE_SIGMA_PER_PIXEL , 1e-4f);
                const float planeWeight = exp(-d / planeSigma);
                float geometryWeight = nW * planeWeight;

                if(baseWeight * geometryWeight > 1e-4f)
                {
                    const float3 rayStart = offsetRay(P , N , generalBuffer.playerPosition);
                    const float3 rayEnd = offsetRay(probe.position , probe.normal , generalBuffer.playerPosition);
                    const float3 toProbe = rayEnd - rayStart;
                    const float probeDistance3D = length(toProbe);
                    if(probeDistance3D > 1e-3f)
                    {
                        geometryWeight *= calculateShadows(rayStart , toProbe / probeDistance3D , probeDistance3D * 0.99f);
                    }
                }

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
                        const float texelWeight = rcTexelWeight1D(texelDistance.x , float(BUILD_SAMPLE_COUNT)) * rcTexelWeight1D(texelDistance.y , float(BUILD_SAMPLE_COUNT));
                        if(texelWeight <= 0.0f) continue;

                        // the corner of the block of the probe + the texel in it
                        const int2 pixel = currentProbe * buildSize + currentBuild;

                        // the direction of the texel of this probe : the texels of the other directions than the one that is looked for count less
                        const float3 texelDir = octahedralDecode((float2(currentBuild) + 0.5f) / float(buildSize));
                        const float directionWeight = pow(saturate(dot(dir , texelDir)) , RC_DIRECTION_POWER);

                        const float4 sampleData = (currentImage ? RCImages[RC::imageIndex(0u , generalBuffer.frameIndex)][uint2(pixel)] : RCImages[RC::prevImageIndex(generalBuffer.frameIndex)][uint2(pixel)]);

                        // the limit of the radiance : the texel on the other side of an edge of the light is skipped , and it does not count in the confidence either ( it is not a
                        // failure of the geometry )
                        if(hasReference && abs(luminance(sampleData.xyz) - referenceLuminance) > MAX_RC_RADIANCE_LIMIT) continue;

                        // what the texel would weigh without the geometry : the denominator of the confidence
                        idealSum += baseWeight * texelWeight * directionWeight;

                        const float weight = baseWeight * geometryWeight * texelWeight * directionWeight;
                        if(weight <= 0.0f) continue;
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

        // how much of the neighbourhood is good : 1 - all of it , 0 - nothing ( every probe is on another surface or does not see the point )
        const float confidence = (idealSum > 0.0f) ? saturate(weightSum / (idealSum * RC_CONFIDENCE_FULL)) : 0.0f;
#if RC_DEBUG_CONFIDENCE
        return float3(confidence , confidence , confidence);
#endif
        if(weightSum <= 0.0f) return 0.0f;

        // the distance of the hits if there are any , of the misses if there are not
        hitDistance = (hitWeight > 0.0f) ? (hitDistanceSum / hitWeight) : (missDistanceSum / weightSum);
        // the fallback is 0 : the light that the weak weights give is not made into a full one by the division
        return (result / weightSum) * confidence;
#else
        // NO spatial reuse : ONE probe ( the nearest one ) and ONE texel of its map ( the nearest one ) , taken as they are
        const int2 nearestProbe = clamp(int2(floor(probePos + 0.5f)) , int2(0 , 0) , probeCount - 1);
        const uint nearestIndex = c.PROBE_OFFSET + uint(nearestProbe.y) * c.PROBE_PER_ROW + uint(nearestProbe.x);
        // a probe in an object or on the sky : there is nothing to take
        if(RCProbeBuffer[nearestIndex].validProbe == 0u) return 0.0f;

        const float2 directionUV = octahedralEncode(dir);
        const int2 nearestTexel = clamp(int2(floor(directionUV * float(buildSize))) , int2(0 , 0) , int2(buildSize - 1 , buildSize - 1));
        const uint2 nearestPixel = uint2(nearestProbe * buildSize + nearestTexel);

        const float4 data = (currentImage ? RCImages[RC::imageIndex(0u , generalBuffer.frameIndex)][nearestPixel] : RCImages[RC::prevImageIndex(generalBuffer.frameIndex)][nearestPixel]);
        bool dataHit;
        float dataDistance;
        RC::unpackHit(data.a , dataHit , dataDistance);
        hitDistance = dataDistance;
        return data.xyz;
#endif
    }

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

// indirect GI
// ReSTIR GI

bool isSpecularMode(const float roughness , const float metallic){ return (metallic >= 0.7f && roughness < 0.4f); }

float3 evalFcos(const GeneralObjectInfo info , const bool specularMode , const float3 L){
    if(specularMode){
        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
        return evalBRDF(info.N , info.V , L , info.roughness , F0) * calcMultiScatterFactor(F0 , info.roughness);
    }
    return evalDiffuse(info.albedo , info.N , L , info.metallic);
}

// how much of the sky the point sees : 0 ( covered , a room ) ... 1 ( open ) . the share of the rays of the cone around the up direction that left the scene . it is a WEIGHT , not a yes / no :
// with a few rays the yes / no flickers at the border of a cover , the weight changes in steps
float skyVisibilityAt(const float3 P , const float3 N , inout uint seed)
{
    const uint sampleCount = features.skyboxSampleCount;
    // no rays : nothing says that the point is covered
    if(sampleCount == 0u) return 1.0f;

    float escaped = 0.0f;
    [loop]
    for(uint i = 0 ; i < sampleCount ; i++){
        const float3 direction = sampleCone(float3(0.0f , 1.0f , 0.0f), 0.5f , seed );
        const float shadow = calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , direction , 100.0f);
        if(shadow == 1.0f) escaped += 1.0f;
    }
    return escaped / float(sampleCount);
}

float3 computeLightAt(GeneralObjectInfo data, inout uint seed){
    // in order to literally minimaze the atmosphere effect , we actually need to lower the chances of it to appear * 
    uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
    uint2 pixel = DispatchRaysIndex().xy;

    findLightIndices(data.P , lightIndices , lightCount);

    const float3 saveAlbedo = max(data.albedo , 1e-20);

    float3 result = 0.0f ;
    if(lightCount){
        // EVERY lamp of the cell of the world grid is added up ( no random choice , so no division by the chance of the choice ) . a lamp that does not see the point is skipped
        // before its light is computed : outside of its radius , behind the surface , or covered ( the shadow ray to the centre of the lamp )
        const float3 V = normalize(-WorldRayDirection());
        data.V = V;
        const float3 rayOrigin = offsetRay(data.P , data.N , generalBuffer.playerPosition);

        for(uint lampSlot = 0u ; lampSlot < lightCount ; lampSlot++)
        {
            const lg::LightSource light = lightBuffer[lightIndices[lampSlot]];

            // the centre of the lamp , not a random point of its sphere ( the indirect light : a big sphere reached through the walls )
            float3 realL = light.position - data.P;
            const float dist = length(realL);
            realL = realL / max(dist , 1e-4f);

            // the cheap tests first : no ray for a lamp that does not reach the point or looks at its back
            const float NdotL = saturate(dot(data.N , realL));
            if(NdotL <= 0.0f) continue;
            const float window = lightAttenuation(light , dist);
            if(window <= 0.0f) continue;

            // the lamp is not seen directly : it is skipped
            const float shadow = calculateShadows(rayOrigin , realL , dist);
            if(shadow <= 0.0f) continue;

            float3 diffuse , specular;
            computeLight(data , realL , diffuse , specular);
            // diffuse /= saveAlbedo;
            result += NdotL * light.diffuse * diffuse * shadow * window;
        }
    }else{
        result =  0.01f;
    }
    const float skyVisibility = skyVisibilityAt(data.P , data.N , seed);

    if(sunBuffer.intensity > 0.0f){
        float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
        // now we also have to check if the shadow actually hits it , since if not, there is no point at computing the light , since its also the calcualtion
        const float3 L = randomInSunRadius(seed , sunBuffer , -sunBuffer.direction);
        // under the open sky the shadow of the sun is traced , in the other case the sun gives nothing ( 0 )
        float sunShadow = 0.0f;
        if(skyVisibility > 0.0f){
            sunShadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , 1000.0f);
        }
        computeLight(data , L , sunDiffuse , sunSpecular);
        result += (sunDiffuse) *  clamp(dot(data.N , L),  0.0f , 1.0f) * sunBuffer.diffuse  * sunShadow;
    }

    // the light of the sky : only under the open sky ( skyVisibility > 0 )
    if(skyVisibility > 0.0f){
        const float3 skyV = normalize(-WorldRayDirection());
        const float3 skyF0 = lerp(float3(0.04f , 0.04f , 0.04f) , data.albedo , data.metallic);
        const float skyNdotV = max(dot(data.N , skyV) , 0.0001f);
        const float3 skyF = F_fresnel(skyF0 , skyNdotV);
        const float3 skyKD = (1.0f - skyF) * (1.0f - data.metallic);
        result += sampleSkybox(data.P , data.N) * skyKD;
    }

    {
        const bool relfective = isSpecularMode(data.roughness , data.metallic);

        if(!relfective){
            float pdf = 0.0f;
            float3 L = 0.0f;
            sampleDiffuseDirection(float2(rand(seed) , rand(seed)) , data.N , L , pdf );
            if(pdf > 0.0f)
            {   
                float hitDistance;
                float3 brdf = evalDiffuse(data.albedo , data.N , L , data.metallic);
                // brdf /= saveAlbedo;
                result += sanitizeRadiance(RC::processRC(L , data.P , data.N , hitDistance , false) , RC::FEEDBACK_MAX_RADIANCE) * brdf / pdf;
            }
        }
        else
        {
            float pdf = 0.0f;
            float3 L = 0.0f;
            sampleSpecularDirection(float2(randNoise() , randNoise()) , data.N , data.V , data.roughness , L , pdf );
            if(pdf > 0.0f && dot(data.V , L) > 0.0f)
            {   
                float hitDistance;
                const float3 F0 = lerp(float3(0.04 , 0.04 , 0.04) , saveAlbedo , data.metallic);
                float3 brdf = evalBRDF(data.N , data.V , L , max(data.roughness , 0.3f) , F0);
                result += sanitizeRadiance(RC::processRC(L , data.P , data.N , hitDistance , false) , RC::FEEDBACK_MAX_RADIANCE) * brdf / pdf;
            }
        }
    }

    return sanitizeRadiance(result , RC::CACHE_MAX_RADIANCE);
}

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

    load.finalColor = computeLightAt(info , load.seed);
    load.alpha = 1.0f;
    load.hitDistance = RayTCurrent();
}

[shader("closesthit")]
void main(inout RayPayLoad load , in BuiltInTriangleIntersectionAttributes attr){
    // skip it . 255 is HIT_KIND_TRIANGLE_BACK_FACE . updatePoses writes validProbe = 1 for every probe at the start of the frame , the test only takes it away
    if(load.mode == 1u){
        if(HitKind() == 255u){
            RCProbeBuffer[RCCascadesBuffer[load.cascadeID].PROBE_OFFSET + DispatchRaysIndex().y].validProbe = 0u;
        }
        load.finalColor = 0.0f;
        load.alpha = 1.0f;
        load.hitDistance = RayTCurrent();
        return;
    }
    initNoise(DispatchRaysIndex().xy , generalBuffer.frameIndex);
    const float3 V = WorldRayDirection();
    GeneralObjectInfo info = readData(attr.barycentrics);
    load.finalColor = computeLightAt(info , load.seed);
    load.hitDistance = RayTCurrent();
    load.alpha = 1.0f;
}