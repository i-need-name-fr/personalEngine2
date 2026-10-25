// the alpha test of a hit ( glTF alphaMode MASK ) : false when the pixel of the diffuse map at the hit is under the cutoff of the material , so the hit does not exist
// ( the any hit shaders call IgnoreHit , the ray queries do not commit the candidate ) . it is the same test as the discard of the fragment shader of the Deferred render
//
// it needs these names to be declared before the include : instanceLayout , meshBuffer , vertices , indices , LUT_Buffer , virtualTextureIndex , allTextures and a sampler that is
// named by ALPHA_TEST_SAMPLER ( #define ALPHA_TEST_SAMPLER typicalSampler )
bool passesAlphaTest(const uint instanceID , const uint primitive , const float2 barycentrics){
    const InstanceData instance = instanceLayout[instanceID];
    const LUT lut = LUT_Buffer[instance.LUT_index];
    // the materials without the test and without a map : everything passes
    if(lut.alphaMode == 0 || lut.diffuseID == NON_VALID_SLOT) return true;

    const MeshData mesh = meshBuffer[instance.meshDataIndex];
    const uint base = mesh.firstIndex + primitive * 3u;
    const uint i0 = mesh.firstVertex + indices[base + 0u];
    const uint i1 = mesh.firstVertex + indices[base + 1u];
    const uint i2 = mesh.firstVertex + indices[base + 2u];

    const float w0 = 1.0f - barycentrics.x - barycentrics.y;
    const float2 uv = vertices[i0].UV * w0 + vertices[i1].UV * barycentrics.x + vertices[i2].UV * barycentrics.y;

    const uint id = virtualTextureIndex[lut.diffuseID];
    const float alpha = allTextures[id].SampleLevel(ALPHA_TEST_SAMPLER , uv , 0).a;
    return alpha >= lut.alphaCutoff;
}
