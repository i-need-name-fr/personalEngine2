#pragma once

#include "Mesh2.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace mox
{
    namespace gl = fastgltf;

    class Model3 final
    {
    public:

        SetTypicalOperators(Model3)
        SetTypicalTransferOperators(Model3)

        struct CreateInfo
        {
            std::filesystem::path path;
            std::filesystem::path personalParallax{NonTexturePath};
            std::filesystem::path personalMetallic{NonTexturePath};
            std::filesystem::path personalRoughness{NonTexturePath};

            // the material that is used when the file has no texture for the mesh ( the same as in Model2 )
            glm::vec3 emission{0.0f};
            glm::vec3 color{0.5f , 0.5f , 0.5f};
            float roughness{1.0f};
            float metallic{0.0f};

            // false ( the default ) : the matrix of every NODE is baked into its vertices , the model has one matrix for all of its meshes ( a modular kit stays as the file
            // lays it out ) . true : the vertices stay in the space of the node ( a part of a kit around its own origin ) , the meshes are grouped by the NAME of the node
            // ( getPart ) , and every part can be put in the scene many times with its own matrices ( Mesh2::addMatrix ) . every mesh still has the instance 0 at the identity
            bool keepNodeSpace{false};

            // true : the metallicRoughness map of the file is a packed ARM map ( the red channel is the occlusion : the maps of Poly Haven , the modular kits ) , so the ao map is
            // taken from the same file . false ( the default ) : the red channel of the map is NOT trusted ( the glTF spec does not use it , Sponza has 0 there : the occlusion 0 turned
            // all the direct light black ) , the ao map is only the occlusionTexture of the material when it has one
            bool armPackedOcclusion{false};
        };

        explicit Model3(const CreateInfo &info)
        {
            const auto result = _implProcessModel(info);
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        ~Model3() noexcept
        {
            processDelete();
        }

        // =================================== INSTANCES ( the same as Model2 ) ===================================

        // a new instance of every mesh of the file
        void addMatrix(const glm::mat4 &matrix) noexcept{
            for(auto& mesh : meshes){
                mesh->addMatrix(matrix);
            }
        }

        // the instance with this index ( 0 is the one that exists from the start ) of every mesh
        void setMatrix(const glm::mat4 &matrix , const uint32_t index = 0) noexcept{
            for(auto& mesh : meshes){
                mesh->updateMatrix(matrix , index);
            }
        }

        void deleteMatrix(const uint32_t index = 0) noexcept{
            for(auto& mesh : meshes){
                mesh->deleteMatrix(index);
            }
        }

        void changeLightStatus(const uint8_t status) noexcept{
            for(auto& mesh : meshes){
                mesh->changeLightStatus(status);
            }
        }

        [[nodiscard]] std::size_t meshCount() const noexcept{
            return meshes.size();
        }

        [[nodiscard]] const std::vector<pMesh2>& getMeshes() const noexcept{
            return meshes;
        }

        // the meshes of the node with this name ( keepNodeSpace ) : a node can have several primitives , every one is a mesh of its own . empty when there is no such node
        [[nodiscard]] const std::vector<pMesh2>& getPart(const std::string &name) const noexcept{
            static const std::vector<pMesh2> empty{};
            const auto it = parts.find(name);
            return it == parts.end() ? empty : it->second;
        }

        [[nodiscard]] const std::unordered_map<std::string , std::vector<pMesh2>>& getParts() const noexcept{
            return parts;
        }

    private:
        
        void processCopy(Model3&& input) noexcept
        {
            meshes = std::move(input.meshes);
            parts = std::move(input.parts);
        }

        void processDelete() noexcept
        {
            meshes.clear();
            parts.clear();
        }

        void _implProcessTextures(Mesh2::CreateInfo &info , const gl::Primitive &prim , const gl::Asset &asset , const std::filesystem::path &directory , const bool armPackedOcclusion) noexcept
        {
            // the uri of a texture in glTF is relative to the folder of the .gltf file ( "textures/x.jpg" ) , not to the working directory of the program
            const auto fullPath = [&directory](const std::filesystem::path &relative) -> std::string{
                return (directory / relative).string();
            };

            if(prim.materialIndex.has_value())
            {   
                info.diffusePath = NonTexturePath;
                info.metallicPath = NonTexturePath;
                info.roughnessPath = NonTexturePath;
                info.normalPath = NonTexturePath;
                info.aoPath = NonTexturePath;
                const gl::Material& mat = asset.materials[prim.materialIndex.value()];
                // diffuse
                if(mat.pbrData.baseColorTexture.has_value())
                {
                    const gl::TextureInfo& texInfo = mat.pbrData.baseColorTexture.value();
                    const auto& tex = asset.textures[texInfo.textureIndex];
                    if(tex.imageIndex.has_value())
                    {
                        const auto& image = asset.images[tex.imageIndex.value()];
                        if(auto* uriSource = std::get_if<gl::sources::URI>(&image.data))
                        {
                            const auto path = uriSource->uri.fspath();
                            info.diffusePath = fullPath(path);
                        }
                    }
                }
                // normal
                if(mat.normalTexture.has_value())
                {
                    const gl::TextureInfo& texInfo = mat.normalTexture.value();
                    const auto& tex = asset.textures[texInfo.textureIndex];
                    if(tex.imageIndex.has_value())
                    {
                        const auto& image = asset.images[tex.imageIndex.value()];
                        if(auto* uriSource = std::get_if<gl::sources::URI>(&image.data))
                        {
                            const auto path = uriSource->uri.fspath();
                            info.normalPath = fullPath(path);
                        }
                    }
                }
                // ambient occlusion
                if(mat.occlusionTexture.has_value())
                {
                    const gl::TextureInfo& texInfo = mat.occlusionTexture.value();
                    const auto& tex = asset.textures[texInfo.textureIndex];
                    if(tex.imageIndex.has_value())
                    {
                        const auto& image = asset.images[tex.imageIndex.value()];
                        if(auto* uriSource = std::get_if<gl::sources::URI>(&image.data))
                        {
                            const auto path = uriSource->uri.fspath();
                            info.aoPath = fullPath(path);
                        }
                    }
                }

                // RM
                if(mat.pbrData.metallicRoughnessTexture.has_value())
                {
                    const gl::TextureInfo& texInfo = mat.pbrData.metallicRoughnessTexture.value();
                    const auto& tex = asset.textures[texInfo.textureIndex];
                    if(tex.imageIndex.has_value())
                    {
                        const auto& image = asset.images[tex.imageIndex.value()];
                        if(auto* uriSource = std::get_if<gl::sources::URI>(&image.data))
                        {
                            const auto path = uriSource->uri.fspath();
                            info.roughnessPath = fullPath(path);
                            info.metallicPath = fullPath(path);
                            // only when the model says that the red channel is the occlusion ( and the material has no occlusionTexture of its own )
                            if(armPackedOcclusion && info.aoPath == NonTexturePath){
                                info.aoPath = fullPath(path);
                            }
                        }
                    }
                }

                // the alpha test ( MASK ) : the pixels of the base colour map under the cutoff do not exist . BLEND is not supported : it is drawn as an opaque one
                if(mat.alphaMode == gl::AlphaMode::Mask)
                {
                    info.alphaMasked = true;
                    info.alphaCutoff = static_cast<float>(mat.alphaCutoff);
                }

                // the factors of the material multiply the maps in glTF , the shaders do not multiply : a map with the factor 0 is not used .
                // rock_moss_set_01 : its metallicRoughness map is the ROUGHNESS map ( the blue channel is the roughness too ) and the metallic factor is 0 ,
                // without this the rock would be a metal with the metallic = the roughness
                if(mat.pbrData.metallicFactor == 0.0f)
                {
                    info.metallicPath = NonTexturePath;
                    info.metallic = 0.0f;
                }
            }   
            else
            {
                info.roughness = 1.0f;
                info.metallic = 1.0f;
                info.color = glm::vec3(0.0f);
            }
        }

        [[nodiscard]] ErrorDataOutput<void> _implProcessMesh(const gl::Mesh &inputMesh , gl::Asset& asset , const CreateInfo &input , const glm::mat4 &world) noexcept
        {   
            for(auto& mesh : inputMesh.primitives)
            {
                std::vector<Vertex> vertices{};
                std::vector<std::uint32_t> indices{};
                if(mesh.type != fastgltf::PrimitiveType::Triangles) continue;
                if(!mesh.indicesAccessor.has_value()) continue;
                const gl::Accessor& indicesAcces = asset.accessors[mesh.indicesAccessor.value()];
                indices.resize(indicesAcces.count);
                gl::iterateAccessorWithIndex<std::uint32_t>(asset , indicesAcces , [&](std::uint32_t data , std::size_t idx) ->void {indices[idx] = data;});

                Mesh2::CreateInfo info{};
                info.needsCalcNormal = false;
                info.emission = input.emission;
                info.color = input.color;
                info.roughness = input.roughness;
                info.metallic = input.metallic;
                // position
                {
                    auto* it = mesh.findAttribute("POSITION");
                    if(it == mesh.attributes.end())
                    {
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create one of the meshes for {} , since the position wasnt defined" , inputMesh.name)};
                        engineLogger(error);
                        continue;
                    }

                    const auto& acc = asset.accessors[it->accessorIndex];
                    vertices.resize(acc.count);
                    gl::iterateAccessorWithIndex<glm::vec3>(asset , acc , [&](glm::vec3 data , std::size_t idx) ->void {vertices[idx].position = data;});
                }

                // normal
                {
                    auto* it = mesh.findAttribute("NORMAL");
                    if(it == mesh.attributes.end())
                    {
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create one of the meshes for {} , since the normal wasnt defined , will be calculated in the mesh" , inputMesh.name)};
                        engineLogger(error);
                        info.needsCalcNormal = true;
                    }
                    else
                    {
                        const auto& acc = asset.accessors[it->accessorIndex];
                        gl::iterateAccessorWithIndex<glm::vec3>(asset , acc , [&](glm::vec3 data , std::size_t idx) ->void {vertices[idx].normal = data;});
                    }
                }
                // UV
                {
                    auto* it = mesh.findAttribute("TEXCOORD_0");
                    if(it == mesh.attributes.end())
                    {
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create one of the meshes for {}, however will set to 0 , since the UV wasnt defined" , inputMesh.name)};
                        engineLogger(error);
                    }
                    else
                    {
                        const auto& acc = asset.accessors[it->accessorIndex];
                        gl::iterateAccessorWithIndex<glm::vec2>(asset , acc , [&](glm::vec2 data , std::size_t idx) ->void {vertices[idx].UV = data;});
                    }
                }

                _implProcessTextures(info , mesh , asset , input.path.parent_path() , input.armPackedOcclusion);
                if(input.personalMetallic != NonTexturePath)
                {
                    info.metallicPath = input.personalMetallic.string();
                }

                if(input.personalRoughness != NonTexturePath)
                {
                    info.roughnessPath = input.personalRoughness.string();
                }
            
                if(input.personalParallax != NonTexturePath)
                {
                    info.depthPath = input.personalParallax.string();
                }
                // the matrix of the NODE ( glTF puts every part of a modular model in its place by the transform of its node ) is baked into the vertices : the model then has
                // one matrix of its own for all the meshes , like the models from OBJ . the normals go with the inverse transpose of the matrix
                {
                    const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(world)));
                    for(Vertex& vertex : vertices){
                        vertex.position = glm::vec3(world * glm::vec4(vertex.position , 1.0f));
                        const glm::vec3 transformed = normalMatrix * vertex.normal;
                        vertex.normal = (glm::length(transformed) > 1e-12f) ? glm::normalize(transformed) : transformed;
                    }

                    // a mirrored node ( a negative determinant ) turns the triangles inside out : the order of the corners is turned over
                    if(glm::determinant(glm::mat3(world)) < 0.0f){
                        for(std::size_t i = 0 ; i + 2 < indices.size() ; i += 3){
                            std::swap(indices[i + 1] , indices[i + 2]);
                        }
                    }
                }
                info.needsCalcTBN = true;
                info.inputVertices = vertices;
                info.inputIndices = indices;

                meshes.push_back(std::make_shared<Mesh2>(info));
            }

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> _implProcessModel(const CreateInfo &input) noexcept
        {
            if(!Parser_.has_value())
            {
                Parser_ = std::move(gl::Parser());
            }

            auto& pars = Parser_.value();
            auto data = gl::GltfDataBuffer::FromPath(input.path);
            if(data.error() != gl::Error::None)
            {
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create Model -> {} , Error -> {}" , input.path.string() , gl::getErrorName(data.error()))};
                return std::unexpected(error);
            }

            const auto options = gl::Options::GenerateMeshIndices | gl::Options::LoadExternalBuffers;
            auto Asset_check = pars.loadGltf(data.get() , input.path.parent_path() , options);

            if(Asset_check.error() != gl::Error::None)
            {
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create Model -> {} , Error -> {}" , input.path.string() , gl::getErrorName(Asset_check.error()))};
                return std::unexpected(error);
            }

            auto asset = std::move(Asset_check.get());

            std::size_t sceneIndex = asset.defaultScene.has_value() ? asset.defaultScene.value() : 0;
            if(!asset.scenes.empty() && sceneIndex < asset.scenes.size())
            {
                ErrorDataOutput<void> status{};
                gl::iterateSceneNodes(asset , sceneIndex , gl::math::fmat4x4() , [&](gl::Node &node , const gl::math::fmat4x4 &matrix) -> void {
                    if(!status.has_value() || !node.meshIndex.has_value()) return;
                    const glm::mat4 world = glm::make_mat4(matrix.data());
                    const std::size_t before = meshes.size();
                    status = _implProcessMesh(asset.meshes[node.meshIndex.value()] , asset , input , input.keepNodeSpace ? glm::mat4(1.0f) : world);

                    // the meshes that this node has made belong to the part of its name
                    const std::string name = node.name.empty() ? std::format("mesh_{}" , node.meshIndex.value()) : std::string(node.name.begin() , node.name.end());
                    for(std::size_t i = before ; i < meshes.size() ; i++){
                        parts[name].push_back(meshes[i]);
                    }
                });
                if(!status.has_value()) return status;
            }
            else
            {
                for(const auto& mesh : asset.meshes)
                {
                    const auto result = _implProcessMesh(mesh , asset , input , glm::mat4(1.0f));
                    if(!result.has_value()) return result;
                }
            }

            return {};
        }
        inline static std::optional<gl::Parser> Parser_{};
        std::vector<pMesh2> meshes{};
        std::unordered_map<std::string , std::vector<pMesh2>> parts{};
    };

    using pModel3 = std::shared_ptr<Model3>;
}