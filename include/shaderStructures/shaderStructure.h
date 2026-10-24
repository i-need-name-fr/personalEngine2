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
// RayTracing pipeline: 3 hit groups, 3 miss shaders
// the SHADOW hit group is an any hit only ( the bodies of the lamps do not shadow ) : the shadow rays use it as RayContributionToHitGroupIndex
#define RT_HITGROUP_GENERAL 0
#define RT_HITGROUP_HITINFO 1
#define RT_HITGROUP_SHADOW  2

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
    uint32_t BLAS_index;

#else

    uint LUT_index;
    uint meshDataIndex;
    uint needsToBeDraw;
    uint BLAS_index;

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
    int32_t isLightSource = 0;
    
    // the packed PBR map ( ARM ) : -1 ( NON_VALID_SLOT ) when there is none
    int32_t armTextureID = -1;
    // the map of the ambient occlusion
    int32_t aoTextureID = -1;

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
    int isLightSource;

    int armTextureID;
    int aoTextureID;

#endif

    float padding0;
    float padding1;
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
    float exposure;

    // what the picture shows : DEFAULT_MODE , ALBEDO_MODE , NORMAL_MODE , DEPTH_MODE , INDIRECT_MODE
    uint32_t viewMode;


    // the balance : the direct light ( lamps , sun ) is multiplied by directWeight , the indirect ( bounces , reflections , the ambient ) by indirectWeight
    float directWeight;
    float indirectWeight;
    // SSDO / SSGI are on : the ambient light is a part of their rays ( the rays that escape bring it ) , PostProcess does not add it again
    uint32_t ambientInIndirect;

    // 1 : the normal maps are OpenGL style ( green up ) , the green channel is flipped when the normal is built ; 0 : DirectX style . appended at the END : it takes the 4 bytes
    // of padding after ambientInIndirect , so the struct is still 592 bytes and the shaders that do not read it stay valid
    uint32_t normalMapOpenGL;


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
    float exposure;

    uint viewMode;

    float directWeight;
    float indirectWeight;
    uint ambientInIndirect;

    uint normalMapOpenGL;


#endif
};


struct LightSource{
#ifdef __cplusplus

    alignas(16) glm::vec3 diffuse;
    alignas(16) glm::vec3 position;

    float lightLinear;
    float lightQuadratic;
    float lightRadius;

    // the size of the lamp itself ( the radius of the light sphere , meters ) : the angle of the cone to the lamp is sourceRadius / distance , it makes the
    // penumbra . lightRadius is NOT it : that is how far the lamp lights . appended at the END : the struct is still 48 bytes ( it was 40 , padded ) , so the shaders
    // that do not read it stay valid
    float sourceRadius;

#else

    float3 diffuse;
    float3 position;

    float lightLinear;
    float lightQuadratic;
    float lightRadius;

    float sourceRadius;

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
    // the light of ALL the candidates ( the lamps of the tile ) , simply summed : diffuse and specular are denoised by different denoisers , so they are two
    alignas(16) glm::vec4 diffuseColor;
    alignas(16) glm::vec4 specularColor;
#else
    uint M;
    uint chosenIndex;
    float W;
    float weightSum;
    float4 diffuseColor;
    float4 specularColor;

    void update(uint lightIndex, float w, float randomValue){
        weightSum += w;
        M += 1u;
       
        if(weightSum > 0.0f && randomValue * weightSum < w){
            chosenIndex = lightIndex;
        }
    }
#endif
};

struct ReservoirGI{
#ifdef __cplusplus
    alignas(16) glm::vec3 radiance;
    alignas(16) glm::vec3 samplePos;
    alignas(16) glm::vec3 sampleNormal;

    uint32_t M;
#else

    float3 radiance;
    float3 samplePos;
    float3 sampleNormal;

    uint M;

    void update( ReservoirGI can ,float w , uint canM , float randomValue){
        M += canM;
        weightSum += w;
        // same guarded form as ReservoirDI::update - no division by a possibly zero weightSum
        if(weightSum > 0.0f && randomValue * weightSum < w){
            samplePos = can.samplePos;
            sampleNormal = can.sampleNormal;
            radiance = can.radiance;
            // the roughness of the sample point travels with the sample : it tells if the sample can be moved to another pixel
            roughness = can.roughness;
        }
    }
#endif

    float W;
    float weightSum;
    float roughness;
};


struct SunSource{
#ifdef __cplusplus

    alignas(16) glm::vec3 diffuse;
    // the size of the sun for the shadow : the half angle of the cone to the sun ( radians ) , the real sun is 0.00465 . the bigger , the softer the penumbra .
    // it sits in the padding after diffuse ( offset 12 ) : the offsets of direction ( 16 ) and intensity ( 28 ) did not move and the struct is still 32 bytes ,
    // so every shader that was compiled before sees the same layout
    float sourceRadius;
    alignas(16) glm::vec3 direction;
    // the strength of the sun ( the scalar of diffuse = color * intensity ) . it sits in the padding after direction ( offset 28 )
    float intensity;

#else

    float3 diffuse;
    float sourceRadius;
    float3 direction;
    float intensity;

#endif
};

struct ForwardPlusInfo{
#ifdef __cplusplus

    uint32_t totalLights;
    uint32_t TILE_SIZE;
    uint32_t TILE_PER_ROW;
    uint32_t TILE_TOTAL_AMOUNT;

    // a copy of GeneralFeatures::lightShadowFade for the Forward+ ( it has no features buffer ) : the lamp goes into the lists of the tiles with the
    // radius enlarged by it , so the lamp is there where its fading shadow still counts . appended at the END
    float lightShadowFade;

#else

    uint totalLights;
    uint TILE_SIZE;
    uint TILE_PER_ROW;
    uint TILE_TOTAL_AMOUNT;

    float lightShadowFade;

#endif
};

#define SSAO_REQUIRED 1
#define HBAO_REQUIRED 2
#define RTAO_REQUIRED 3
#define RSSAO_REQUIRED 4
#define RHBAO_REQUIRED 5
#define AO_NOT_REQUIRED 0

#define SSR_REQUIRED 1

#define NON_REQUIRED 0
#define SSGI_REQUIRED 2
#define DDGI_REQUIRED 3

#define DEFAULT_MODE 0
#define ALBEDO_MODE 1
#define NORMAL_MODE 2
#define DEPTH_MODE 3
#define METALLIC_MODE 4
#define ROUGHNESS_MODE 5

// the indirect light alone ( SSDO / SSGI / DDGI / SSR ) , without the direct light
#define INDIRECT_MODE 6

// the ambient occlusion after the bilateral filter ( 1 = nothing covers the pixel , darker = more covered )
#define AO_MODE 7

// the shadow after the bilateral filter ( 1 = lit , darker = in the shadow , red = more than 1 ) : the number that PostProcess multiplies the direct light by
#define SHADOW_MODE 8

// how many modes the R key cycles through ( 0 .. SHADOW_MODE )
#define TOTAL_VIEW_MODES 9

struct GeneralFeatures{
#ifdef __cplusplus

    uint32_t AO;
    uint32_t AO_iteration;
    float AOradius;
    float AO_intens;

    uint32_t indirectMode;

    uint32_t VSM;
    uint32_t RayQuery;

    // the number of the rays of the soft shadows : of the lamp and of the sun ( 0 - that source does not trace )
    uint32_t RayQueryLightSampleCount;
    float shadowThickness;
    float temporalShadowK;
    float light_sun_Proportion;

    uint32_t RT;
    uint32_t PT;

    uint32_t ReSTIRGI;
    uint32_t ReSTIRGI_temporal;
    uint32_t ReSTIRGI_spatial;

    uint32_t RadianceCascade;

    uint32_t currentMode;

    uint32_t randomValues;
    uint32_t volumentricFog;

    float IndirectShare;
    float maxDepth;

    uint32_t RayQuerySunSampleCount;

    // ReSTIR DI of the LightPass : 1 - the best lamp of the tile is chosen ( the shadows go to it ) , 0 - all the lamps of the tile are added up .
    // it is the last one : the fields before it keep their offsets
    uint32_t ReSTIR;

    // how far beyond its radius the shadow of a lamp still counts , in the radii of the lamp : the shadow fades out between lightRadius and
    // lightRadius * ( 1 + lightShadowFade ) ( see lightShadowReach ) . 0 - the shadow ends at the radius with a jump
    // appended at the END : the fields before it keep their offsets , so the shaders that do not read it stay valid
    float lightShadowFade;

    // how many directions of the hemisphere the sky light ( ambient ) is averaged over per pixel and frame ( the denoiser and the history do the rest ) . 0 - none
    uint32_t skyboxSampleCount;

    // the shadow of the sun gets weaker with the distance to what covers the pixel ( along the ray to the sun ) : 0 at the covering object , 1 at this distance
    // ( smoothstep ) . 0 - no fade , the shadow is 0 everywhere in the shade ( RT only , it needs the closest hit of the shadow ray )
    float sunShadowFade;

    // how dark a shadow can get : PostProcess puts the shadow of the direct light between this ( full shadow ) and 1 ( lit ) . 0 - the shadow is black
    float shadowMinimum;
#else

    uint AO;
    uint AO_iteration;
    float AOradius;
    float AO_intens;

    uint indirectMode;

    uint VSM;
    uint RayQuery;

    uint RayQueryLightSampleCount;
    float shadowThickness;
    float temporalShadowK;
    float light_sun_Proportion;

    uint RT;
    uint PT;

    uint ReSTIRGI;
    uint ReSTIRGI_temporal;
    uint ReSTIRGI_spatial;

    uint RadianceCascade;

    uint currentMode;

    uint randomValues;
    uint volumentricFog;

    float IndirectShare;
    float maxDepth;

    uint RayQuerySunSampleCount;

    uint ReSTIR;

    float lightShadowFade;

    uint skyboxSampleCount;

    float sunShadowFade;

    float shadowMinimum;

#endif
};

struct FogVolume{
#ifdef __cplusplus

    alignas(16) glm::vec3 center;
    alignas(16) glm::vec3 halfExtent;
    alignas(16) glm::vec3 albedo;

    float density;
    float feather;
    float g;
    float gridSize;
    float heightFallOff;
    float fogBaseHeight;

    // the own glow of the fog ( the radiance that it scatters without any light : the color of the fog that is seen far away ) . the albedo is another thing :
    // it is the color that the fog gives to the light of the lamps and of the sun .
    // NO alignas ( 16 ) : after the float the compiler of the shader puts a float3 right behind it ( offset 68 : three floats fit into the rest of the 16 byte row ) ,
    // alignas would move it to 80 and the shader would read the padding
    glm::vec3 color;

#else

    float3 center;
    float3 halfExtent;
    float3 albedo;

    float density;
    float feather;
    float g;
    float gridSize;
    float heightFallOff;
    float fogBaseHeight;

    float3 color;

#endif
};

struct SkyViewData{
    float cameraHeight;
    float sunCosZenith;
    // the azimuth of the sun in the world ( the horizontal direction of the sun : x = cos , z = sin ) . the sky view LUT always has the sun at the azimuth 0 ,
    // the cube turns the LUT by this angle
    float sunCosAzimuth;
    float sunSinAzimuth;
};

// new Probes
namespace DDGI{

    struct ProbeInfo{
#ifdef __cplusplus

        alignas(16) glm::vec3 position;

#else

        float3 position;

#endif
    };

    struct ProbeTileInfo{
#ifdef __cplusplus
        alignas(16) glm::vec3 position;

        uint32_t totalProbes;
        uint32_t firstProbe;
        
#else

        float3 position;
        uint totalProbes;
        uint firstProbe;

#endif
        float padding0;
        float padding1;
    };

    struct GeneralProbeInfo{
#ifdef __cplusplus
        uint32_t TOTAL_PROBE_TILES_COUNT;
        uint32_t TOTAL_PROBES_COUNT;
        uint32_t PROBE_RESOLUTION;
        uint32_t PROBES_PER_X;

        uint32_t PROBE_TILE_SIZE;
        uint32_t ATLAS_X;
        uint32_t ATLAS_Y;
#else
        uint TOTAL_PROBE_TILES_COUNT;
        uint TOTAL_PROBES_COUNT;
        uint PROBE_RESOLUTION;
        uint PROBES_PER_X;

        uint PROBE_TILE_SIZE;
        uint ATLAS_X;
        uint ATLAS_Y;
#endif
        float padding0;
    };
}

namespace RC{

    // valid probe tells us , if we can even reach it , if no , it means that it might be in object , and we dont want to trace rays for it 
    struct ProbeInfo{
#ifdef __cplusplus

        alignas(16) glm::vec3 position;
        uint32_t cascadeID;
        uint32_t validProbe;
#else

        float3 position;
        uint cascadeID;
        uint validProbe;

#endif
        float padding0;

        // appended at the END ( the probe is placed on a surface of the screen , RCplacement.hlsl ) : the normal of the surface and the distance from the camera , they are
        // for the weights of the merge ( a probe on another surface / far in depth must not give its light )
#ifdef __cplusplus
        alignas(16) glm::vec3 normal;
#else
        float3 normal;
#endif
        float cameraDistance;
    };

    struct CascadeInfo{
#ifdef __cplusplus
        uint32_t PROBES_PER_CASCADE;
        // the probes of the cascade are a 3D grid of GRID_SIZE ^ 3 nodes ( the index of a node is x + GRID_SIZE * ( y + GRID_SIZE * z ) )
        uint32_t BUILD_SIZE;
        uint32_t PROBE_RAY_COUNT;
        uint32_t CASCADE_X;

        uint32_t CASCADE_Y;   
        uint32_t PROBE_OFFSET;
        uint32_t BLOCK_SIZE;
        uint32_t PROBE_PER_ROW;
#else
        uint PROBES_PER_CASCADE;
        uint BUILD_SIZE;
        uint PROBE_RAY_COUNT;
        uint CASCADE_X;
        
        uint CASCADE_Y;   
        uint PROBE_OFFSET;
        uint BLOCK_SIZE;
        uint PROBE_PER_ROW;

#endif  
        // the distance between two nodes of the grid ( meters ) . the grid is snapped to it : the nodes stay at the same places of the world while the player moves
        float HISTORY_COUNT;
        // the interval of the rays of the cascade along the ray from its probe : [ INTERVAL_START , INTERVAL_END ] ( meters ) . the end of a cascade is the start of the next one
        float INTERVAL_START;
        float INTERVAL_END;
        float PROBE_RESOLUTION;
    };

    static const float MAX_RC_HISTORY = 16ul;

    struct GeneralProbeInfo{
#ifdef __cplusplus
        uint32_t TOTAL_CASCADE_COUNT;
        uint32_t TOTAL_PROBES;
#else
        uint TOTAL_CASCADE_COUNT;
        uint TOTAL_PROBES;
#endif
        float padding0;
        float padding1;
    };
}


/*
#ifdef __cplusplus

#else

#endif
*/

// new world light grid
//  INFO : basically i dont want to make forward plus , but instead i want to make a BHV system for world space 
namespace lg{

    // basically 32.0f is enough for our grid , since we dont want to render smth out 
    #ifdef __cplusplus
        // grid size is how far in each direction we can see the light , X Y Z 
        inline static constexpr float GRID_SIZE = 32.0f;
        // each grid is a position , and that position has its own size , for example we will be using 8.0f
        // more than enough i think , but if its needed , i will do more
        inline static constexpr float GRID_CELL_SIZE = 4.0f;
        inline static constexpr float GRID_CELL_OFFSET = 18.0f;
        inline static constexpr uint32_t MAX_LIGHT_PER_CELL = 64ul;
    #else
        static const float GRID_SIZE = 32.0f;
        static const float GRID_CELL_SIZE = 4.0f;
        static const uint MAX_LIGHT_PER_CELL = 64ul;
        static const float GRID_CELL_OFFSET = 18.0f;

    #endif

    #ifdef __cplusplus
        inline static constexpr uint32_t MAX_LIGHT_SOURCES = 4096ul;
    #else
        static const uint MAX_LIGHT_SOURCES = 4096ul;
    #endif


    struct LightSource{
    #ifdef __cplusplus

        alignas(16) glm::vec3 diffuse;
        alignas(16) glm::vec3 position;
    #else
        float3 diffuse;
        float3 position;
    #endif
        float Linear;
        float Quadratic;
        float radius;

        float sourceRadius;
    };

    // idea is like this , there is a new Buffer , LightBufferIndices
    // main goal is to minimaze the loop for finding the suitable light , instead we will be using BVH algorithm
    #ifdef __cplusplus
        using LightSourceIndex = int32_t;
    #else
        typedef int LightSourceIndex;
    #endif

    struct LightCell{
    #ifdef __cplusplus

        alignas(16) glm::vec3 position;
        uint32_t totalLightCount;
        int32_t lightIndices[MAX_LIGHT_PER_CELL];
        
    #else

        float3 position;
        uint totalLightCount;
        uint lightIndices[MAX_LIGHT_PER_CELL];

    #endif

        float padding0;
        float padding1;
    };

    struct GridCascadeInfo{
    #ifdef __cplusplus
        uint32_t totalLightCells;
        int32_t totalLightCount;
    #else
        uint totalLightCells;
        uint totalLightCount;
    #endif
        float padding0;
        float padding1;        
    };
}

#ifdef __cplusplus
    inline static uint32_t TILE_SIZE = 16;
#endif

#ifndef __cplusplus

// ---- stateful RNG ---------------------------------------------------------
// PCG32. The state is advanced on every draw, so two draws from the same
// variable are independent. Never derive two values from the same seed value.

static const uint  AP_WIDTH  = 32; // screen X
static const uint  AP_HEIGHT = 32; // screen Y
static const uint  AP_SLICES = 32; // дистанция (froxel-слои)

static const float DEPTH_RANGE = 128000.0; // метров — максимальная дистанция froxel'а
static const uint FACE_SIZE = 128;
static const float PLANET_RADIUS     = 6371000.0;
static const float ATMOSPHERE_RADIUS = 6471000.0;
static const float H_RAYLEIGH = 8500.0;
static const float H_MIE      = 1200.0;
static const float3 BETA_RAYLEIGH_SCAT = float3(5.8e-6, 1.35e-5, 3.31e-5);
static const float  BETA_MIE_SCAT      = 4.0e-5;
static const float  BETA_MIE_ABS       = 4.4e-6;
static const uint   AP_STEPS = 20;

static const uint LUT_WIDTH  = 192;
static const uint LUT_HEIGHT = 108;

// the z depth ( meters ) that the froxel volume of the fog covers , the slices are spread logarithmically between these two . fogCalc ( writes the slices ) and
// PostProcess ( reads them ) have to use the same numbers
static const float FOG_NEAR = 0.01f;
static const float FOG_FAR  = 200.0f;

static const uint SKY_VIEW_STEPS = 32;

static const float OFFSET_PER_CASCADE = 10.0f;

// the temporal accumulation of the shadow is NOT endless : the second channel of the shadow image is the number of the frames that the pixel has gathered ( 1 .. MAX_SHADOW_FRAME ) ,
// the weight of the new frame is 1 / ( the number + 1 ) , so after MAX_SHADOW_FRAME frames the old ones fade out
static const uint MAX_SHADOW_FRAME = 12u;

// a history that is far from this frame ( the camera has jerked , the shadow behind the pixel is another one ) must not stay : the difference up to this value is the noise of
// the rays and does not matter , above it the number of the frames is cut down , and at the difference 1 the history is thrown away completely ( the new frame has the weight 1 )
static const float SHADOW_HISTORY_TOLERANCE = 0.35f;

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

// the same , but with the origin given by the caller : WorldRayOrigin() is NOT allowed in a ray generation shader ( VUID-WorldRayOriginKHR-04431 ) ,
// so a raygen passes the position of the camera : offsetRay(P , N , generalBuffer.playerPosition)
float3 offsetRay(float3 P , float3 N , float3 origin){
    const float scale = 0.001f + 0.003f * length(P - origin);
    return P + N * scale;
}

// the shadow of a lamp does not end at its radius with a jump : it fades out between lightRadius and lightRadius * ( 1 + fade ) , where fade is
// GeneralFeatures::lightShadowFade ( ImGui : the shadow part of the features ) . 1 inside of the radius , 0 at the end of the fade . Forward+ puts a lamp
// into the lists of the tiles with the same enlarged radius ( ForwardPlusInfo::lightShadowFade ) , so the lamp is there where its fading shadow still counts
float lightShadowReach(const float distance , const float lightRadius , const float fade){
    const float outerRadius = lightRadius * (1.0f + max(fade , 0.0f));
    // no fade : the shadow ends at the radius ( smoothstep with equal edges divides by zero )
    if(outerRadius <= lightRadius) return (distance <= lightRadius) ? 1.0f : 0.0f;
    return 1.0f - smoothstep(lightRadius , outerRadius , distance);
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

// randNoise : the numbers for the DIRECTIONS ( the error is mostly high frequency in the picture and evenly spread in time , so the denoisers and the
// temporal accumulation have less to remove than with white noise ) . It is NOT for the choices of a reservoir ( rand * weightSum < w ) : those must stay
// random , the neighbours are not allowed to repeat each other there
// every invocation starts with initNoise( pixel , frameIndex ) . the state is private to the entry point : a hit shader has to call it itself
static uint2 g_noisePixel = uint2(0u , 0u);
static uint g_noiseFrame = 0u;
static uint g_noiseDimension = 0u;

void initNoise(uint2 pixel , uint frameIndex){
    g_noisePixel = pixel;
    g_noiseFrame = frameIndex;
    g_noiseDimension = 0u;
}

// interleaved gradient noise ( Jimenez ) : the values of the neighbours are far from each other , without a texture
float interleavedGradientNoise(float2 p){
    return frac(52.9829189f * frac(dot(p , float2(0.06711056f , 0.00583715f))));
}

float randNoise(){
    const uint d = g_noiseDimension++;
    const float2 p = float2(g_noisePixel);

    // in the picture : the dimensions 0..3 are the pattern , swapped and mirrored ( they do not repeat each other , the test gave a correlation 0 ) .
    // the next ones are a hash of the pixel : not blue any more , but still independent
    float spatial;
    if(d == 0u)      spatial = interleavedGradientNoise(p);
    else if(d == 1u) spatial = interleavedGradientNoise(p.yx);
    else if(d == 2u) spatial = interleavedGradientNoise(float2(-p.x , p.y));
    else if(d == 3u) spatial = interleavedGradientNoise(float2(p.y , -p.x));
    else{
        uint h = g_noisePixel.x * 1973u + g_noisePixel.y * 9277u + d * 26699u + 1u;
        spatial = float(rngNext(h) >> 8u) * (1.0f / 16777216.0f);
    }

    // in time : the R2 sequence ( the plastic number ) , in integers , so it is exact for any frame . the even and the odd dimensions are the two
    // coordinates of it , so a pair ( the two numbers of a direction ) covers the square evenly over the frames
    const uint alpha = ((d & 1u) == 0u) ? 3242174889u : 2447445414u;
    const float temporal = float((g_noiseFrame * alpha) >> 8u) * (1.0f / 16777216.0f);

    return frac(spatial + temporal);
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

// the same direction , but from randNoise() ( initNoise() first ) instead of the seed
float3 randomInHemisphere(float3 normal){
    const float rand1 = randNoise();
    const float rand2 = randNoise();

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

// the same half vector , but from randNoise() ( initNoise() first ) instead of the seed
float3 GGX_halfVector(float3 normal , float roughness){
    // two separate statements : the order of the arguments of a constructor is not promised
    const float rand1 = randNoise();
    const float rand2 = randNoise();
    return GGX_halfVectorFromXi(float2(rand1 , rand2) , normal , roughness);
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

// the distance along a ray ( dir is a unit vector ) from a point INSIDE of a sphere to the point where the ray leaves it . 0 if the ray does not meet the sphere .
// Radiance Cascade : the interval of a ray of a cascade ends on the sphere of the next cascade , and the combine pass continues the ray from that point
float raySphereExit(float3 origin , float3 dir , float3 center , float radius){
    const float3 oc = origin - center;
    const float b = dot(oc , dir);
    const float c = dot(oc , oc) - radius * radius;
    const float discriminant = b * b - c;
    if(discriminant < 0.0f) return 0.0f;
    return max(-b + sqrt(discriminant) , 0.0f);
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

// the alpha of the images of Radiance Cascade ( RCImages , rgba16f ) holds TWO things in the 16 bits of the half float :
//   the SIGN BIT  : the ray ended in a hit ( 1 ) or not ( 0 ) , the sky and the ray that left its interval are not a hit
//   the other 15 bits : the distance to the hit , meters ( the half float of the absolute value , it is exact enough up to MAX_HIT_DISTANCE )
// the cleared image ( 0 ) is " no hit , distance 0 " . the packed value is NOT a number to average : unpack the samples first , then mix the colours , the hit flags
// ( the coverage ) and the distances separately
namespace RC{

    // the half float ends at 65504
    static const float MAX_HIT_DISTANCE = 60000.0f;
    static const float MAX_RADIANCE_TOLERANCE = 0.35f;

    float packHit(const bool hit , const float distance){
        const float d = clamp(distance , 0.0f , MAX_HIT_DISTANCE);
        // the sign is set by the bits : it works for the distance 0 too ( -0.0 )
        return asfloat(asuint(d) | (hit ? 0x80000000u : 0u));
    }

    void unpackHit(const float packed , out bool hit , out float distance){
        hit = (asuint(packed) & 0x80000000u) != 0u;
        distance = abs(packed);
    }

    // which image of RCImages[] belongs to a cascade : the cascade 0 has TWO images ( ping pong : the images 0 and 1 ) , the cascades 1 .. N - 1 have one image each ( the images
    // 2 .. N ) . so the chain of a frame is  0 2 3 4 5  or  1 2 3 4 5 , it depends on the parity of the frame
    // imageIndex : the image that the cascade is written to / read from in THIS frame , prevImageIndex : the image of the cascade 0 of the LAST frame
    uint imageIndex(const uint cascadeID , const uint frameIndex){
        return (cascadeID == 0u) ? (frameIndex & 1u) : (cascadeID + 1u);
    }

    uint prevImageIndex(const uint frameIndex){
        return (frameIndex + 1u) & 1u;
    }
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