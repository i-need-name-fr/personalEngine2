#pragma once

#include "Mesh2.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace mox{

    class Model2;
    using pModel2 = std::shared_ptr<Model2>;

    // =============================== INFO ===============================
    // Model2 loads a file through assimp and makes one Mesh2 per assimp mesh .
    // it is only a loader : the instances , the BLAS and the buffers are all handled by Mesh2 ,
    // so Mesh2::Begin() has to be called before the model is created and Mesh2::End() after all of them
    // ( the same rule as for a single Mesh2 )
    //
    // NO RAGDOLL here : a model is a static group of meshes , every mesh of the file gets the same matrix
    // the first instance ( index 0 ) already exists at the origin , since every Mesh2 creates it in its own constructor
    // =============================== END ===============================

    class Model2 final{
    public:

        struct CreateInfo{
            std::string path{};

            // material , used when the file has no texture for the mesh
            glm::vec3 emission{0.0f};
            glm::vec3 color{0.5f , 0.5f , 0.5f};
            float roughness{1.0f};
            float metallic{0.0f};

            uint8_t isALightObject{false};
        };

        Model2() = delete;
        Model2(Model2& input) = delete;
        Model2(const Model2& input) = delete;
        Model2& operator=(Model2& input) = delete;
        Model2& operator=(const Model2& input) = delete;

        Model2(Model2&& input) noexcept{
            performCopy(std::move(input));
        }

        Model2& operator=(Model2&& input) noexcept{
            if(this == &input) return *this;
            performCopy(std::move(input));
            return *this;
        }

        ~Model2() = default;

        explicit Model2(const CreateInfo &info){
            const auto result = processModel(info);
            if(!result.has_value()){
                engineLogger(result.error());
                THROW_MESSAGE;
            }

            const ErrorDataType success{MOX_ERROR_TYPE_SUCCESS , std::format("successfully created the model -> {} , meshes -> {}" , info.path , meshes.size())};
            engineLogger(success);
        }

        // =================================== INSTANCES ===================================

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

        [[nodiscard]] const std::vector<std::shared_ptr<Mesh2>>& getMeshes() const noexcept{
            return meshes;
        }

    private:

        // shared_ptr : a Mesh2 is created in place and never moved
        std::vector<std::shared_ptr<Mesh2>> meshes{};
        std::string directory{};

        void performCopy(Model2&& input) noexcept{
            meshes = std::move(input.meshes);
            directory = std::move(input.directory);
        }

        // a name that can be compared : lower case , spaces and underscores are the same , one kind of slash
        [[nodiscard]] static std::string normalizeKey(std::string name) noexcept{
            for(auto& c : name){
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if(c == ' ') c = '_';
                if(c == '\\') c = '/';
            }
            return name;
        }

        // the file `name` in the folder `folder` : as it is written , and if there is no such file , a file of the folder with the same
        // name in any letter case and with spaces / underscores swapped ( an MTL of Blender names the image datablock , not the file )
        [[nodiscard]] static std::string resolveIn(const std::filesystem::path &folder , std::string name){
            namespace fs = std::filesystem;
            std::error_code ec;

            std::replace(name.begin() , name.end() , '\\' , '/');
            if(name.empty() || name[0] == '*') return NonTexturePath;   // '*' = a texture embedded in the file

            const fs::path asWritten = folder / name;
            if(fs::exists(asWritten , ec)) return asWritten.generic_string();

            const std::string wanted = normalizeKey(fs::path(name).filename().generic_string());
            for(const auto& entry : fs::directory_iterator(folder.empty() ? fs::path(".") : folder , ec)){
                std::error_code entryError;
                if(!entry.is_regular_file(entryError)) continue;
                if(normalizeKey(entry.path().filename().generic_string()) == wanted) return entry.path().generic_string();
            }
            return NonTexturePath;
        }

        [[nodiscard]] std::string resolveFile(std::string name) const{
            std::error_code ec;
            std::replace(name.begin() , name.end() , '\\' , '/');
            if(std::filesystem::path(name).is_absolute() && std::filesystem::exists(name , ec)) return name;
            return resolveIn(std::filesystem::path(directory.empty() ? "." : directory) , name);
        }

        // the texture of the material for the first of the types that has one . an MTL does not use the types the way the names
        // suggest ( see processMesh ) , so several types are tried . a name that is written but does not lead to a file is logged
        [[nodiscard]] std::string findTexturePath(const aiMaterial* material , std::initializer_list<aiTextureType> types) const{
            if(!material) return NonTexturePath;

            for(const aiTextureType type : types){
                if(material->GetTextureCount(type) == 0) continue;

                aiString str{};
                if(material->GetTexture(type , 0 , &str) != AI_SUCCESS) continue;

                const std::string resolved = resolveFile(str.C_Str());
                if(resolved != NonTexturePath) return resolved;

                const ErrorDataType warning{MOX_ERROR_TYPE_WARNING , std::format("the material of a model names the texture |{}| , but there is no such file in -> {}" , str.C_Str() , directory)};
                engineLogger(warning);
            }
            return NonTexturePath;
        }

        // the maps that the MTL does not name : next to "Foo_Color.png" there is often "Foo_Normal.png" , "Foo_Roughness.png" ...
        [[nodiscard]] std::string findSiblingMap(const std::string &diffusePath , std::initializer_list<const char*> suffixes) const{
            namespace fs = std::filesystem;
            if(diffusePath == NonTexturePath) return NonTexturePath;

            const fs::path diffuse(diffusePath);
            const std::string stem = diffuse.stem().generic_string();
            const std::size_t cut = stem.find_last_of("_ ");
            if(cut == std::string::npos) return NonTexturePath;

            const std::string tail = normalizeKey(stem.substr(cut + 1));
            if(tail != "color" && tail != "diffuse" && tail != "albedo" && tail != "basecolor" && tail != "base") return NonTexturePath;

            const std::string base = stem.substr(0 , cut);
            for(const char* suffix : suffixes){
                for(const char* extension : {".png" , ".jpg" , ".jpeg" , ".tga" , ".bmp"}){
                    const std::string found = resolveIn(diffuse.parent_path() , base + "_" + suffix + extension);
                    if(found != NonTexturePath) return found;
                }
            }
            return NonTexturePath;
        }

        // an empty pointer in the value means "this assimp mesh is skipped" ( no triangles ) , it is not an error
        [[nodiscard]] ErrorDataOutput<std::shared_ptr<Mesh2>> processMesh(const aiMesh* mesh , const aiScene* scene , const CreateInfo &data) noexcept{
            if(!mesh) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to process the mesh of a model , since the assimp mesh is nullptr"});

            Mesh2::CreateInfo info{};

            info.inputVertices.reserve(mesh->mNumVertices);
            for(uint32_t i = 0 ; i < mesh->mNumVertices ; i++){
                Vertex vertex{};
                vertex.position = glm::vec3(mesh->mVertices[i].x , mesh->mVertices[i].y , mesh->mVertices[i].z);

                if(mesh->HasNormals()){
                    vertex.normal = glm::vec3(mesh->mNormals[i].x , mesh->mNormals[i].y , mesh->mNormals[i].z);
                }

                if(mesh->mTextureCoords[0]){
                    vertex.UV = glm::vec2(mesh->mTextureCoords[0][i].x , mesh->mTextureCoords[0][i].y);
                }

                info.inputVertices.push_back(vertex);
            }

            for(uint32_t face = 0 ; face < mesh->mNumFaces ; face++){
                const aiFace &currentFace = mesh->mFaces[face];
                // Triangulate makes everything a triangle , lines and points that are left are not drawn
                if(currentFace.mNumIndices != 3) continue;
                for(uint32_t i = 0 ; i < 3 ; i++){
                    info.inputIndices.push_back(currentFace.mIndices[i]);
                }
            }

            if(info.inputVertices.empty() || info.inputIndices.empty()){
                const ErrorDataType warning{MOX_ERROR_TYPE_WARNING , std::format("a mesh of the model -> {} has no triangles , it is skipped" , directory)};
                engineLogger(warning);
                return std::shared_ptr<Mesh2>{nullptr};
            }

            // Mesh2 computes the tangents itself ( from the indices and the UVs , a mirrored part gets a flipped frame ) , the ones of assimp
            // are not read : they depend on whether they were computed before or after the flip of the UVs
            info.needsCalcTBN = true;

            info.emission = data.emission;
            info.color = data.color;
            info.roughness = data.roughness;
            info.metallic = data.metallic;
            info.isALightObject = data.isALightObject;

            info.diffusePath = NonTexturePath;
            info.normalPath = NonTexturePath;
            info.depthPath = NonTexturePath;
            info.roughnessPath = NonTexturePath;
            info.metallicPath = NonTexturePath;

            if(mesh->mMaterialIndex < scene->mNumMaterials){
                const aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
                // what an OBJ / MTL file ( Blender ) calls what :
                //   map_Kd -> DIFFUSE      map_Bump / bump -> HEIGHT ( a normal map in a Blender export ) , norm -> NORMALS
                //   disp -> DISPLACEMENT   map_Ns -> SHININESS ( the roughness image )   map_refl -> REFLECTION ( the metallic image )
                //   map_Pr / map_Pm are the PBR extension ( DIFFUSE_ROUGHNESS / METALNESS )
                info.diffusePath = findTexturePath(material , {aiTextureType_DIFFUSE , aiTextureType_BASE_COLOR});
                info.normalPath = findTexturePath(material , {aiTextureType_NORMALS , aiTextureType_NORMAL_CAMERA , aiTextureType_HEIGHT});
                info.depthPath = findTexturePath(material , {aiTextureType_DISPLACEMENT});
                info.roughnessPath = findTexturePath(material , {aiTextureType_DIFFUSE_ROUGHNESS , aiTextureType_SHININESS});
                info.metallicPath = findTexturePath(material , {aiTextureType_METALNESS , aiTextureType_REFLECTION});

                // the MTL did not name some maps : look next to the color map ( Foo_Color.png -> Foo_Normal.png ... )
                if(info.normalPath == NonTexturePath) info.normalPath = findSiblingMap(info.diffusePath , {"Normal" , "NormalGL" , "Nrm"});
                if(info.depthPath == NonTexturePath) info.depthPath = findSiblingMap(info.diffusePath , {"Displacement" , "Height" , "Depth" , "Disp"});
                if(info.roughnessPath == NonTexturePath) info.roughnessPath = findSiblingMap(info.diffusePath , {"Roughness" , "Rough"});
                if(info.metallicPath == NonTexturePath) info.metallicPath = findSiblingMap(info.diffusePath , {"Metallic" , "Metalness" , "Metal"});
            }

            try{
                return std::make_shared<Mesh2>(info);
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create a Mesh2 for the model -> {} ( {} ) , check the logs" , directory , e.what())});
            }
        }

        [[nodiscard]] ErrorDataOutput<void> processNode(const aiNode* node , const aiScene* scene , const CreateInfo &data) noexcept{
            for(uint32_t i = 0 ; i < node->mNumMeshes ; i++){
                auto result = processMesh(scene->mMeshes[node->mMeshes[i]] , scene , data);
                if(!result.has_value()) return std::unexpected(result.error());
                if(result.value()) meshes.push_back(std::move(result.value()));
            }

            for(uint32_t i = 0 ; i < node->mNumChildren ; i++){
                const auto result = processNode(node->mChildren[i] , scene , data);
                if(!result.has_value()) return std::unexpected(result.error());
            }
            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> processModel(const CreateInfo &data) noexcept{
            if(data.path.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create model , since the path is empty"});
            }

            Assimp::Importer importer;
            const aiScene* scene = importer.ReadFile(data.path.c_str() , aiProcess_GenSmoothNormals | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs);
            if(!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to load model {} , assimp error -> {}" , data.path , importer.GetErrorString())});
            }

            const std::size_t slash = data.path.find_last_of("/\\");
            directory = (slash == std::string::npos) ? "" : data.path.substr(0 , slash);

            return processNode(scene->mRootNode , scene , data);
        }
    };

}
