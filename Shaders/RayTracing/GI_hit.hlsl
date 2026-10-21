#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RaytracingAccelerationStructure TLAS;
[[vk::binding(1 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(2 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(3 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(4 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(5 , 0)]] ConstantBuffer<SunSource> sunBuffer;

[[vk::binding(6 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(7 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(8 , 0)]] RWStructuredBuffer<lg::LightSource> lightBuffer;

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


struct [raypayload] HitRayPayLoad{
    bool   didHit : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 worldPos : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 normal : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 albedo : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float  roughness : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float  metallic : write(caller,closesthit , miss) : read(caller,closesthit , miss);
};

// set 1 : the textures of the TextureManager2 ( the same set as the Deferred render ) . virtualTextureIndex turns the index of the LUT into the physical one
[[vk::binding(0 , 1)]] StructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState typicalSampler;
[[vk::binding(2 , 1)]] Texture2D allTextures[];

// the closest hit of the GI rays : it fills the payload with what the shader of the G buffer would have written for this surface ( the same
// material , the same normal map , the same parallax ) , so the secondary hit looks like the primary one

// the normal maps are OpenGL style ( green up ) or DirectX style ( green down ) : generalBuffer.normalMapOpenGL ( ImGui ) , THE SAME FLAG as in DeferredRender/fragmentShader.hlsl

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
    const float depth = allTextures[physicalIndex].SampleLevel(typicalSampler , currentUV , 0).x;
    const float height = 1.0f - depth;
    const float height_scale = 0.001f;
    const float2 p = (V.xy) / V.z * height * height_scale;
    return currentUV - float2(p.x , -p.y);
}

[shader("closesthit")]
void main(inout HitRayPayLoad load , in BuiltInTriangleIntersectionAttributes attr){
    // InstanceID() is the instanceCustomIndex of the TLAS instance = the index in the instance layouts
    const InstanceData instance = instanceLayout[InstanceID()];
    const uint meshID = instance.meshDataIndex;
    const LUT lut = LUT_Buffer[instance.LUT_index];

    const HitSurface surface = interpolateHit(meshID , PrimitiveIndex() , attr.barycentrics);

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
        // OpenGL maps ( green up ) are used as they are with this TBN , the DirectX ones ( green down ) are flipped . ( it was the other way round : every map was inverted )
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
        metallic = allTextures[id].SampleLevel(typicalSampler , uv , 0).x;
    }

    float roughness = lut.roughness;
    if(lut.roughnessTextureID != NON_VALID_SLOT){
        const uint id = virtualTextureIndex[lut.roughnessTextureID];
        roughness = allTextures[id].SampleLevel(typicalSampler , uv , 0).x;
    }

    // every field of the payload is written : the caller reads all of them
    if(dot(normal , -WorldRayDirection()) < 0.0f) normal = -normal;
    load.didHit = true;
    load.worldPos = hitPoint;
    load.normal = normal;
    load.albedo = albedo;
    load.roughness = roughness;
    load.metallic = metallic;
}
