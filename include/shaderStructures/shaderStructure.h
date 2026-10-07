#ifndef SHADER_HPP
#define SHADER_HPP

/*
#ifdef __cplusplus

#else

#endif
*/

#ifdef __cplusplus

    #include <glm/glm.hpp>
    #include <glm/gtc/type_ptr.hpp>
    #include <glm/gtc/matrix_transform.hpp>

#endif


struct Vertex{
#ifdef __cplusplus

    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 normal;
    alignas(16) glm::vec3 tangent;
    alignas(16) glm::vec3 bitangent;
    // 64
    alignas(16) glm::vec2 UV;
    
#else

    float3 position;
    float3 normal;

    float3 tangent;
    float3 bitangent;

    float2 UV;

#endif
};




// this all for LUT
#define TEXTURE_REQUIRED 1lu
#define COLOR_REQUIRED 0u

#define MAX_LIGHTS 64

// WE WILL BE CLEANING THE MOTION VECTOR IMAGE EVERY FRAME , if its non RT or PT 
#define SKYBOX_PIXEL 2.0f
#define COLOR_PIXEL 1.0f
#define UNDEFINED_PIXEL 0.0f

#define MAX_AO_ITERATION 64
static const float PI = 3.14159265f;
#define MAX_INSTANCE 15
#define MAX_RAY_CAST 10
#define MAX_LIGHT_SOURCE 200
// ---------------------------------------------------------------------------
// Shader binding table layout.
// These MUST match the order in which the groups are written into the SBT on
// the C++ side. A wrong index makes TraceRay() dispatch into a shader that
// expects a different payload type.
// ---------------------------------------------------------------------------
// RayTracing pipeline: 2 hit groups, 3 miss shaders
#define RT_HITGROUP_GENERAL 0
#define RT_HITGROUP_HITINFO 1

#define RT_MISS_GENERAL 0
#define RT_MISS_HITINFO 1
#define RT_MISS_SHADOW  2

// PathTracing pipeline: 1 hit group, 2 miss shaders
#define PT_HITGROUP_GENERAL 0

#define PT_MISS_GENERAL 0
#define PT_MISS_SHADOW  1

#define PT_REQUIRED 1
#define PT_NOT_REQUIRED 0

#define SHADING_RATE_1x1 0    // без флагов = 1x1
#define SHADING_RATE_1x2 1    // Vertical2Pixels
#define SHADING_RATE_1x4 2    // Vertical4Pixels
#define SHADING_RATE_2x1 4    // Horizontal2Pixels
#define SHADING_RATE_2x2 5    // Vertical2Pixels | Horizontal2Pixels (1|4)
#define SHADING_RATE_2x4 6    // Vertical4Pixels | Horizontal2Pixels (2|4)
#define SHADING_RATE_4x1 8    // Horizontal4Pixels
#define SHADING_RATE_4x2 9    // Vertical2Pixels | Horizontal4Pixels (1|8)
#define SHADING_RATE_4x4 10   

struct InstanceData{
#ifdef __cplusplus

    uint32_t LUT_index;
    uint32_t meshDataIndex;
    uint32_t needsToBeDraw;
    float padding0;

#else

    uint LUT_index;
    uint meshDataIndex;
    uint needsToBeDraw;
    float padding0;

#endif
};



struct MeshData{
#ifdef __cplusplus

    alignas(16) glm::vec3 boxMin;
    alignas(16) glm::vec3 boxMax;

    alignas(16) glm::vec3 boxCenter;
    float boxRadius;
    

    uint32_t firstVertex;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t instanceIndex;

#else

    float3 boxMin;
    float3 boxMax;

    float3 boxCenter;
    float boxRadius;
    

    uint firstVertex;
    uint firstIndex;
    uint indexCount;
    uint instanceIndex;

#endif
};

struct LUT{
#ifdef __cplusplus

    alignas(16) glm::vec3 color;
    alignas(16) glm::vec3 emission;

    float roughness = 0.0f;
    float metallic = 0.0f;

    int32_t diffuseTextureID = 0;
    int32_t normalTextureID = 0;

    int32_t depthTextureID = 0;
    int32_t roughnessTextureID = 0;
    int32_t metallicTextureID = 0;
    int32_t flag = 0;
    
#else

    float3 color;
    float3 emission;

    float roughness;
    float metallic;
    int diffuseID;
    int normalID;

    int depthID;
    int roughnessTextureID;
    int metallicTextureID;
    int flag;

#endif
};

// this is for TextureInfo , since there are times when textures are deleted
#define NON_VALID_SLOT -1

// alignas(16) glm::vec3

struct GeneralInfo{
#ifdef __cplusplus

    glm::mat4 inverseProj;
    glm::mat4 inverseView;
    glm::mat4 projOnly;
    glm::mat4 viewOnly;
    glm::mat4 viewProj; 
    glm::mat4 invViewProj;
    glm::mat4 prevViewProj;
    glm::mat4 viewSkybox;

    alignas(16) glm::vec3 playerPosition;
    alignas(16) glm::vec3 playerDirection;

    float dt;
    uint32_t screenWidth;
    uint32_t screenHeight;
    uint32_t lightCount;
    uint32_t frameIndex;

    float playerFar;
    float playerNear;

    // what the picture shows : DEFAULT_MODE , ALBEDO_MODE , NORMAL_MODE , DEPTH_MODE , INDIRECT_MODE
    uint32_t viewMode;

    // the screen space shadows ( copied from GeneralFeatures )
    uint32_t screenShadow;
    uint32_t shadowSteps;
    float shadowThickness;
    float shadowBias;

    // the balance : the direct light ( lamps , sun ) is multiplied by directWeight , the indirect ( bounces , reflections , the ambient ) by indirectWeight
    float directWeight;
    float indirectWeight;
    // SSDO / SSGI are on : the ambient light is a part of their rays ( the rays that escape bring it ) , PostProcess does not add it again
    uint32_t ambientInIndirect;

    // the cone of the shadow rays ( copied from GeneralFeatures , the angle in radians )
    uint32_t shadowSamples;
    float shadowConeAngle;

#else

    float4x4 inverseProj;
    float4x4 inverseView;
    float4x4 projOnly;
    float4x4 viewOnly;
    float4x4 viewProj;
    float4x4 invViewProj;
    float4x4 prevViewProj;
    float4x4 viewSkybox;

    float3 playerPosition;
    float3 playerDirection;

    float dt;

    uint screenWidth;
    uint screenHeight;
    uint lightCount;
    uint frameIndex;

    float playerFar;
    float playerNear;

    uint viewMode;

    uint screenShadow;
    uint shadowSteps;
    float shadowThickness;
    float shadowBias;

    float directWeight;
    float indirectWeight;
    uint ambientInIndirect;

    uint shadowSamples;
    float shadowConeAngle;

#endif
};

struct LensParams{
#ifdef __cplusplus
    glm::vec2 outputSize;
    uint32_t ghostCount;
    float ghostDispersal;
    float chromaticAberration;
    float lightVisibility;
    // only the pixels brighter than this ( luminance of the final picture , 0..1 ) make ghosts
    float brightThreshold;
    // pads the struct to 32 bytes ( a uniform buffer is read in blocks of 16 )
    float padding;
#else
    float2 outputSize;
    uint ghostCount;
    float ghostDispersal;
    float chromaticAberration;
    float lightVisibility;
    float brightThreshold;
    float padding;
#endif
};

struct LightSource{
#ifdef __cplusplus

    alignas(16) glm::vec3 diffuse;
    alignas(16) glm::vec3 position;

    float lightLinear;
    float lightQuadratic;
    float lightRadius;

#else

    float3 diffuse;
    float3 position;

    float lightLinear;
    float lightQuadratic;
    float lightRadius;

#endif
};

struct VSM_params{
#ifdef __cplusplus
    alignas(16) glm::mat4 viewSpaceMatrix;
    alignas(16) glm::vec3 playerPosition;

    float baseDistance;
    
    uint32_t virtualResolution;
    uint32_t pageSize;
    uint32_t maxClips;

    uint32_t width;
    uint32_t height;
    uint32_t poolWidthInPages;
    uint32_t physicalPoolResolution;
#else

    float4x4 viewSpaceMatrix;
    float3 playerPosition;
    
    float baseDistance;
    uint virtualResolution;
    uint pageSize;
    uint maxClips;

    uint width;
    uint height;

    uint poolWidthInPages;
    uint physicalPoolResolution;

#endif
};


struct ReservoirDI{
#ifdef __cplusplus
    uint32_t M;
    uint32_t chosenIndex;
    float W;
    float weightSum;
#else
    uint M;
    uint chosenIndex;
    float W;
    float weightSum;

    void update(uint lightIndex, float w, float randomValue){
        weightSum += w;
        M += 1u;
       
        if(weightSum > 0.0f && randomValue * weightSum < w){
            chosenIndex = lightIndex;
        }
    }
#endif
};


struct SunSource{
#ifdef __cplusplus

    alignas(16) glm::vec3 diffuse;
    alignas(16) glm::vec3 direction;

#else

    float3 diffuse;
    float3 direction;

#endif
};

struct ProbeInfo{
#ifdef __cplusplus

    alignas(16) glm::vec3 worldPosition;

#else

    float3 worldPosition;

#endif
};

struct ProbeTileInfo{
#ifdef __cplusplus

    alignas(16) glm::vec3 generalPosition;
    uint32_t probesCount;
    uint32_t firstProbe;

#else

    float3 generalPosition;
    uint probesCount;
    uint firstProbe;

#endif
};

struct GeneralProbeTileInfo{
#ifdef __cplusplus
    uint32_t totalProbeTiles;
    uint32_t tileSize;

    uint32_t probesPerRow;
    uint32_t probeResolution;

    uint32_t atlasWidth;
    uint32_t atlasHeight;
#else

    uint totalProbeTiles;
    uint tileSize;
    
    uint probesPerRow;
    uint probeResolution;

    uint atlasWidth;
    uint atlasHeight;
#endif
};

struct ForwardPlusInfo{
#ifdef __cplusplus

    uint32_t totalLights;
    uint32_t TILE_SIZE;
    uint32_t TILE_PER_ROW;
    uint32_t TILE_TOTAL_AMOUNT;
    
#else

    uint totalLights;
    uint TILE_SIZE;
    uint TILE_PER_ROW;
    uint TILE_TOTAL_AMOUNT;

#endif
};

#define SSAO_REQUIRED 1
#define HBAO_REQUIRED 2
#define AO_NOT_REQUIRED 0

#define SSR_REQUIRED 1

#define NON_REQUIRED 0
#define SSDO_REQUIRED 1
#define SSGI_REQUIRED 2
#define DDGI_REQUIRED 3

#define DEFAULT_MODE 0
#define ALBEDO_MODE 1
#define NORMAL_MODE 2
#define DEPTH_MODE 3
// the indirect light alone ( SSDO / SSGI / DDGI / SSR ) , without the direct light
#define INDIRECT_MODE 4

struct GeneralFeatures{
#ifdef __cplusplus

    uint32_t AO;
    uint32_t AO_iteration;
    float AOradius;

    uint32_t SSR;
    uint32_t DDGI;
    uint32_t SSDO;
    uint32_t SSGI;    

    uint32_t VSM;
    uint32_t PCSS;

    uint32_t RT;
    uint32_t PT;

    uint32_t currentMode;
    uint32_t Bloom;
    // the screen space lens flare ( SSLens ) , optional
    uint32_t Lens;

    // screen space shadows of the lamps ( LightPass ) : a ray from the pixel to the lamp is marched through the depth of the screen
    uint32_t ScreenShadow;
    uint32_t ShadowSteps;
    float ShadowThickness;     // how far behind a visible surface a ray still counts as hidden by it
    float ShadowBias;          // the start of the ray is moved along the normal by this
    uint32_t ShadowSamples;    // how many times the shadow is cast : the rays of the cone around the direction to the lamp
    float ShadowConeAngle;     // the half angle of that cone ( degrees ) : the size of the lamp , the width of the soft edge

    // the share of the indirect light in the picture ( 0.6 : 60% indirect , 40% direct )
    float IndirectShare;

#else

    uint AO;
    uint AO_iteration;
    float AOradius;

    uint SSR;
    uint DDGI;
    uint SSDO;
    uint SSGI;    

    uint VSM;
    uint PCSS;

    uint RT;
    uint PT;

    uint currentMode;
    uint Bloom;
    uint Lens;

    uint ScreenShadow;
    uint ShadowSteps;
    float ShadowThickness;
    float ShadowBias;
    uint ShadowSamples;
    float ShadowConeAngle;

    float IndirectShare;

#endif
};

struct BloomParameters{

#ifdef __cplusplus
    float thresHold;
    float radius;
    float intensity;
    uint32_t mipCount;

    uint32_t bloomRequired;
    float gamma;
    float exposure;
    float dt;
#else
    float thresHold;
    float radius;
    float intensity;
    int mipCount;

    bool bloomRequired;
    float gamma;
    float exposure;
    float dt;
#endif
};

#ifdef __cplusplus
    inline static uint32_t TILE_SIZE = 16;
#else   



#endif

#ifndef __cplusplus

// ---- stateful RNG ---------------------------------------------------------
// PCG32. The state is advanced on every draw, so two draws from the same
// variable are independent. Never derive two values from the same seed value.

struct GeneralObjectInfo{
    
    float3 P;
    float3 N;
    float3 V;
    float3 albedo;
    
    float roughness;
    float metallic;

};


float3 offsetRay(float3 P , float3 N){
    const float scale = 0.001f + 0.003f * length(P - WorldRayOrigin());
    return P + N * scale;
}

float luminance(float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }


// rand() returns [0,1) so this can never produce activeLightCount().

uint bufferIndex(uint2 pixel , uint width , uint height){
    const uint2 clamped = min(pixel , uint2(width - 1u , height - 1u));
    return clamped.x + clamped.y * width;
}


uint rngNext(inout uint state){
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// returns [0,1) - never exactly 1.0, so uint(rand() * count) can never index
// one element past the end of an array.
float rand(inout uint state){
    return float(rngNext(state)) * (1.0f / 4294967296.0f);
}

uint initSeed(uint2 pixel, uint frameIndex){
    uint seed = pixel.x * 1973u + pixel.y * 9277u + frameIndex * 26699u + 1u;
    rngNext(seed);
    return seed;
}

float D_GGX(float NdotH , float roughness){
    const float a = roughness * roughness;
    const float a2 = a * a;   // was a * 2 - that is not the GGX alpha squared
    const float denum = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(denum * denum * PI , 0.0001f);
}

float3 F_fresnel(float3 F0 , float VdotH){
    return F0 + (1.0f - F0) * pow(1.0f - VdotH , 5.0f);
}

float G1_schlick(float NdotX , float k){
    return NdotX / (NdotX * (1.0f - k) + k);
}

float G_smith(float NdotV , float NdotL , float roughness){
    const float k_ambient = (roughness + 1) * (roughness + 1 ) / 8.0f;
    const float G1 = G1_schlick(NdotV , k_ambient);
    const float G2 = G1_schlick(NdotL , k_ambient);
    return G1 * G2;
}

// HLSL has no built-in inverse() (unlike GLSL) - that is not a gap in your
// IDE, the intrinsic genuinely does not exist. What a normal actually needs
// is inverse-transpose(M), not inverse(M) itself, and that is cheaper to
// build directly: its ROWS are the pairwise cross products of M's rows,
// scaled by 1/det. This is the standard "normal matrix" trick and avoids a
// general Gauss-Jordan inverse entirely. Correct for non-uniform scale;
// still gives the right direction (and sign, under mirrored/negative-scale
// transforms) after normalize().
float3x3 normalMatrix3x3(float3x3 m){
    const float3 r0 = m[0];
    const float3 r1 = m[1];
    const float3 r2 = m[2];

    const float3 c0 = cross(r1 , r2);
    const float3 c1 = cross(r2 , r0);
    const float3 c2 = cross(r0 , r1);

    const float det = dot(r0 , c0);
    // guard a degenerate (zero-scale / non-invertible) instance instead of
    // dividing by zero
    const float invDet = 1.0f / (abs(det) > 1e-8f ? det : 1e-8f);

    return float3x3(c0 , c1 , c2) * invDet;
}

void buildONB(float3 N , out float3 tangent , out float3 bitangent){
    const float3 up = (abs(N.y) < 0.999f ? float3(0.0f , 1.0f , 0.0f) : float3(1.0f , 0.0f , 0.0f));
    tangent = normalize(cross(up , N));
    bitangent = cross(N , tangent);
}

float3 randomInHemisphere(inout uint seed , float3 normal){
    const float rand1 = rand(seed);
    const float rand2 = rand(seed);

    const float phi = 2.0f * PI * rand2;
    const float cosTheta = sqrt(rand1);
    const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    float3 tangent , bitangent;
    buildONB(normal , tangent , bitangent);
    return localDir.x * tangent + localDir.y * bitangent + localDir.z * normal;
}

// the i-th point of a Hammersley set of `count` points : the same points every time , evenly spread over the unit square ( no random )
float2 Hammersley(uint i , uint count){
    return float2((float(i) + 0.5f) / float(count) , float(reversebits(i)) * 2.3283064365386963e-10f);
}

// the same GGX half vector as GGX_halfVector , but from a given point xi of the unit square instead of two random numbers
float3 GGX_halfVectorFromXi(float2 xi , float3 normal , float roughness){
    const float phi = 2.0f * PI * xi.x;
    float a = roughness * roughness;
    float a2 = a * a ;
    const float cosTheta = sqrt(saturate((1.0f - xi.y) / max(1.0f + (a2 - 1.0f) * xi.y , 1e-6f)));
    const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    float3 tangent , bitangent;
    buildONB(normal , tangent , bitangent);
    return normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * normal);
}

float3 GGX_halfVector(inout uint seed , float3 normal , float roughness){
    const float rand1 = rand(seed);
    const float rand2 = rand(seed);

    const float phi = 2.0f * PI * rand1;   // was rand2 - phi and cosTheta shared one draw
    float a = roughness * roughness;
    float a2 = a * a ;
    const float cosTheta = sqrt(saturate((1.0f - rand2) / max(1.0f + (a2 - 1.0f) * rand2 , 1e-6f)));
    const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    float3 tangent , bitangent;
    buildONB(normal , tangent , bitangent);
    return normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * normal);
}

float3 coneSamplingVector(inout uint seed , float3 L , float d, float r){
    const float rand1 = rand(seed);
    const float rand2 = rand(seed);

    const float phi = 2.0f * PI * rand2;
    const float sinAlpha = (r / max(d , r));
    const float cosAlpha = sqrt(saturate(1.0f - sinAlpha * sinAlpha));
    const float cosTheta = lerp(cosAlpha , 1.0f , rand1);
    const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    float3 tangent , bitangent;
    buildONB(L , tangent , bitangent);
    return normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * L);
}

float3 specularSamplerWeight(float3 F , float G , float3 N , float3 V , float3 L){
    const float3 H = normalize(V + L);
    const float NdotV = max(dot(N , V) , 0.0001f);
    const float VdotH = max(dot(V , H) , 0.0001f);
    const float NdotH = max(dot(N , H) , 0.0001f);
    return F * G * VdotH / max(NdotH * NdotV , 0.0001f);
}

float3 computeLightNonMultiscatter(GeneralObjectInfo info, float3 L){
    // unclamped dots make F_fresnel blow up and G1_schlick divide by zero on
    // back facing geometry
    const float3 H = normalize(info.V + L);
    const float NdotL = max(dot(info.N , L) , 0.0001f);
    const float NdotH = max(dot(info.N , H) , 0.0001f);
    const float NdotV = max(dot(info.N , info.V) , 0.0001f);
    const float VdotH = max(dot(info.V , H) , 0.0001f);

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f), info.albedo , info.metallic);
    const float D = D_GGX(NdotH , info.roughness);
    const float3 F = F_fresnel(F0 , VdotH);
    const float G = G_smith(NdotV , NdotL , info.roughness);

    float3 directSpecular = (D * G * F) / max(4.0f * NdotL * NdotV , 0.0001f);

    const float3 kD = (1.0f - F) * (1.0f - info.metallic);
    const float3 diffuse = kD * info.albedo / PI;
    return (directSpecular + diffuse) * NdotL;
}

float2 octahedralEncode(float3 dir){
    dir /= (abs(dir.x) + abs(dir.y) + abs(dir.z));
    float2 result = dir.xy;
    if(dir.z < 0.0f){
        result = (1.0f - abs(dir.yx)) * sign(dir.xy);
    }
    return result * 0.5f + 0.5f;

}

// the exact inverse of octahedralEncode : uv in [0 , 1] -> a unit direction
float3 octahedralDecode(float2 uv){
    const float2 f = uv * 2.0f - 1.0f;
    float3 n = float3(f.x , f.y , 1.0f - abs(f.x) - abs(f.y));
    // the lower hemisphere is folded over the corners of the square
    const float t = saturate(-n.z);
    n.x += (n.x >= 0.0f) ? -t : t;
    n.y += (n.y >= 0.0f) ? -t : t;
    return normalize(n);
}

float3 sphericalFibonacci(uint i , uint numSamples){
    float phi = 2.0 * PI * frac(i * 0.618034f);   // 0.618034 - золотое сечение (или его обратное)
    float cosTheta = 1.0 - (2.0 * i + 1.0) / (2.0 * numSamples);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// `texture` was an unused parameter, and the X term used * instead of %
uint2 getProbePixelCoord(uint probeIndex , float2 UV , uint probeResolution, uint textureWidth){
    const uint probePerRow = max(textureWidth / probeResolution , 1u);
    const uint2 probeOrigin = uint2(probeIndex % probePerRow , probeIndex / probePerRow) * probeResolution;
    return probeOrigin + (uint2)(UV * (float)probeResolution);
}

float LinearizeDepth(float ndcDepth, float near, float far)
{
    return (near * far) / (far - ndcDepth * (far - near));
}

static const float2 PoissonDisk[16] = { float2(-0.94201624, -0.39906216), float2( 0.94558609,
                                         -0.76890725), float2(-0.09418410, -0.92938870),
                                          float2( 0.34495938, 0.29387760), float2(-0.91588581, 0.45771432),
                                           float2(-0.81544232, -0.87912464), float2(-0.38277543, 0.27676845),
                                            float2( 0.97484398, 0.75648379), float2( 0.44323325, -0.97511554),
                                             float2( 0.53742981, -0.47373420), float2(-0.26496911, -0.41893023),
                                              float2( 0.79197514, 0.19090188), float2(-0.24188840, 0.99706507),
                                               float2(-0.81409955, 0.91437590), float2( 0.19984126, 0.78641367),
                                                float2( 0.14383161, -0.14100790) };

#endif



#ifdef __cplusplus
    static_assert(alignof(Vertex) == 16, "Vertex alignment mismatch");
    static_assert(sizeof(Vertex) == 16 * 5 , "Vertex size must be around 48");
    static_assert(offsetof(Vertex , position) == 0 , "Vertex size must be around 48");
    static_assert(offsetof(Vertex , normal) == 16 , "offset for Vertex from 0 to normal must be around 16");
    static_assert(sizeof(LUT) % 16 == 0);
    static_assert(sizeof(LUT) % 16 ==0 , "");

#endif

#endif