#include "../../include/ShaderStructures/ShaderStructure.h"

// ONE descriptor set for the whole mesh pass ( task + mesh + fragment ) : the SAME declarations in every shader of the pass ,
// they have to match MeshDeferredRender::processInit ( the numbers are the indices of the layout )
// a shader that does not use a binding does not get it in the SPIR-V , so the unused ones cost nothing

// set 0
[[vk::binding(0 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(1 , 0)]] RWStructuredBuffer<InstanceData> instanceLayoutBuffer;
[[vk::binding(2 , 0)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(3 , 0)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(4 , 0)]] RWStructuredBuffer<LUT> LUT_Buffer;
[[vk::binding(5 , 0)]] RWStructuredBuffer<ms::GeometryData> geometryBuffer;
[[vk::binding(6 , 0)]] RWStructuredBuffer<ms::Meshlet> meshletBuffer;
[[vk::binding(7 , 0)]] RWStructuredBuffer<ms::MeshletBound> boundBuffer;
[[vk::binding(8 , 0)]] RWStructuredBuffer<uint> meshletVertices;
[[vk::binding(9 , 0)]] RWStructuredBuffer<uint> meshletIndices;
[[vk::binding(10 , 0)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(11 , 0)]] Texture2D prevDepth;

// set 1 : the texture manager
[[vk::binding(0 , 1)]] StructuredBuffer<uint> virtualTextureIndex;
[[vk::binding(1 , 1)]] SamplerState samp;
[[vk::binding(2 , 1)]] Texture2D allTextures[];
