#pragma once

#include "../vulkanContext/vulkanBase.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../Buffer2/GeneralBuffer.hpp"
#include "../AS2/BLAS2.hpp"

namespace mox{

    inline static constexpr std::string NonTexturePath{"we34ujtijg"};

    template<typename Start , typename... Arg>
    requires (std::is_same_v<Start , Arg> && ...)
    void VectorCat(std::vector<Start> &input , std::vector<Arg>... arg) noexcept{
        (input.insert(input.end() , arg.begin() , arg.end()) , ...);
    }

    class Mesh2 final{
    public:

        inline static pGeneralBuffer2<BufferType::Device , false> vertexBuffer{nullptr};
        inline static pGeneralBuffer2<BufferType::Device , false> indicesBuffer{nullptr};
        inline static pGeneralBuffer2<BufferType::Device , true> instanceBuffer{nullptr};
        inline static pGeneralBuffer2<BufferType::Device , true> instanceLayoutBuffer{nullptr};
        inline static pGeneralBuffer2<BufferType::Device , true> indirectCmdBuffer{nullptr};
        inline static pGeneralBuffer2<BufferType::Device , true> LUT_buffer{nullptr};
        inline static pGeneralBuffer2<BufferType::Device , true> MeshBuffer{nullptr};

        inline static uint32_t totalCmdCount =0;

        inline static std::vector<Vertex> totalVertices{};
        inline static std::vector<uint32_t> totalIndices{};
        inline static std::vector<glm::mat4> instances{};
        inline static std::vector<LUT> LUTs{};
        inline static std::vector<MeshData> totalMeshesData{};
        inline static std::vector<InstanceData> totalInstanceLayouts{};

        inline static std::vector<pBLAS2> BLAS_storage{};
        inline static std::vector<BLAS2_layout> AS_layouts{};

        inline static constexpr uint32_t maxInstanceCount = 50000ul;

        [[nodiscard]] inline static  ErrorDataOutput<void> Begin(VulkanContext* context , TextureManager2* manager)noexcept{
            std::lock_guard<std::mutex> guard(m);
            if(!manager) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to begin recording , since the manager is null"});
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager , context);
                if(!result.has_value()) return result;
            }

            if(endRecording) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_MEDIUM , "failed to begin recording , since the recording is over"});

            Mesh2::context = context;
            Mesh2::manager = manager;

            startedRecording = true;
            return {};
        }

        [[nodiscard]] inline static ErrorDataOutput<void> End() noexcept {
            std::lock_guard<std::mutex> guard(m);
            try{
                // this function creates all the required buffers

                // vertex
                {
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(Vertex) * totalVertices.size();
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

                    vertexBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , translateToSpan(totalVertices));
                }

                // indices
                {
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(uint32_t) * totalIndices.size();
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

                    indicesBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , translateToSpan(totalIndices));
                }

                // instance data
                {
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(InstanceData) * maxInstanceCount;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    instanceLayoutBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }

                // mesh data
                {
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(MeshData) * maxInstanceCount;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    MeshBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }

                // instance
                {
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(glm::mat4) * maxInstanceCount;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    instanceBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }

                // LUT
                {
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(LUT) * maxInstanceCount;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    LUT_buffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }

                // indirect 
                {
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(VkDrawIndexedIndirectCommand) * maxInstanceCount;
                    info.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    indirectCmdBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }

                // BLAS
                {
                    for(auto& instance : totalInstanceLayouts){
                        const uint32_t id = instance.BLAS_index;

                        if(id >= AS_layouts.size()){
                            std::cerr << "somehow, during the creation of BLAS in End() , the index of BLAS layout is bigger then the AS_Layout size\n";
                            continue;
                        }
                        // we want to create the BLAS
                        // or add it if its not created
                        if(id >= BLAS_storage.size()){

                            while(true){
                                BLAS_storage.push_back(nullptr);
                                if(BLAS_storage.size() > id) break;
                            }

                            BLAS2::CreateInfo info{};
                            info.context = context;
                            info.indicesBuffer = indicesBuffer.get();
                            info.vertexBuffer = vertexBuffer.get();
                            info.layout = AS_layouts[id];

                            BLAS_storage[id] = std::make_shared<BLAS2>(info);

                        }else if(!BLAS_storage[id]){

                            BLAS2::CreateInfo info{};
                            info.context = context;
                            info.indicesBuffer = indicesBuffer.get();
                            info.vertexBuffer = vertexBuffer.get();
                            info.layout = AS_layouts[id];

                            BLAS_storage[id] = std::make_shared<BLAS2>(info);

                        }

                    }
                }

                updateMeshData();

                startedRecording = false;
                endRecording = true;
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to End the recording of meshes, please check the logs above"});
            }
            return {};
        }

        inline static const uint8_t getUpdateStatus() noexcept{
            return needsUpdate;
        }

        inline static void performDelete() noexcept{
            vertexBuffer = nullptr;
            indicesBuffer = nullptr;
            indirectCmdBuffer = nullptr;
            instanceBuffer = nullptr;
            BLAS_storage.clear();
            instanceLayoutBuffer = nullptr;
            LUT_buffer = nullptr;
            MeshBuffer = nullptr;
            context = nullptr;
            manager = nullptr;
        }

        inline static void updateMeshData() noexcept{
            if(!needsUpdate) return;
            LUT_buffer->recordData(translateToSpan(LUTs));
            MeshBuffer->recordData(translateToSpan(totalMeshesData));
            instanceBuffer->recordData(translateToSpan(instances));
            instanceLayoutBuffer->recordData(translateToSpan(totalInstanceLayouts));

            // setting the cmd buffer
            std::vector<VkDrawIndexedIndirectCommand> cmds{};
            uint32_t i = 0u; 
            for(auto& instance: totalInstanceLayouts){
                auto mesh = totalMeshesData[instance.meshDataIndex];
                if(!instance.needsToBeDraw) continue ;
                VkDrawIndexedIndirectCommand cmd{};

                cmd.firstIndex = mesh.firstIndex;
                cmd.firstInstance  = i;
                cmd.instanceCount = 1;
                cmd.indexCount = mesh.indexCount;
                cmd.vertexOffset = mesh.firstVertex;

                i++;
                cmds.push_back(cmd);
            }
            totalCmdCount = cmds.size();
            indirectCmdBuffer->recordData(translateToSpan(cmds));

            needsUpdate = false;
        }

        Mesh2() = delete;
        Mesh2(const Mesh2& input) = delete;
        Mesh2(Mesh2& input) = delete;
        Mesh2& operator=(const Mesh2& input) = delete;
        Mesh2& operator=(Mesh2& input) = delete;

        Mesh2(Mesh2&& input) noexcept{
            processCopy(std::move(input));
        }

        Mesh2& operator=(Mesh2&& input) noexcept{
            processCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            std::vector<Vertex> inputVertices{};
            std::vector<uint32_t> inputIndices{};

            std::string diffusePath{NonTexturePath};
            std::string normalPath{NonTexturePath};
            std::string depthPath{NonTexturePath};
            std::string roughnessPath{NonTexturePath};
            std::string metallicPath{NonTexturePath};

            glm::vec3 color{0.0f};
            glm::vec3 emission{0.0f};

            uint8_t isALightObject{false};

            float roughness{0.0f};
            float metallic{0.0f};

            uint8_t needsCalcTBN{true};
        };

        explicit Mesh2(CreateInfo &info){
            // these only read the info of this mesh
            if(info.inputVertices.empty()) throw std::runtime_error("failed to create mesh2 , since the vertices are empty");
            if(info.inputIndices.empty()) throw std::runtime_error("failed to create mesh2 , since the indices are nullptr");
            if(info.inputIndices.size() % 3 != 0) throw std::runtime_error("failed to create mesh2 , since the count of indices should be count % 3 == 0");

            {
                std::lock_guard<std::mutex> guard(m);
                // Begin() and End() change these under the same lock , so they are read under it too
                // End() clears startedRecording too , so the "over" check has to come first or a late mesh gets the wrong message
                if(endRecording) throw std::runtime_error("couldnt add more meshes, since the recording is over");
                if(!startedRecording) throw std::runtime_error("before making meshes , please use Begin()");

                const auto result = processInit(info);
                if(!result.has_value()){
                    engineLogger(result.error());
                    processDelete();
                    THROW_MESSAGE;
                }
            }
        }

        void addMatrix(const glm::mat4 matrix) noexcept{
            std::lock_guard<std::mutex> guard(m);
            addMatrixLocked(matrix);
        }

    private:

        // the work of addMatrix without the lock : the constructor holds it already when processInit adds the first instance ( std::mutex
        // is not recursive , taking it twice in one thread is a deadlock )
        void addMatrixLocked(const glm::mat4 matrix) noexcept{
            const uint32_t instanceID = instances.size();
            const uint32_t meshID = totalMeshesData.size();
            const uint32_t instanceInfoID = totalInstanceLayouts.size();

            instances.push_back(matrix);

            auto aabb = GetAABBCorners(localBoxMin , localBoxMax);
            glm::vec3 boxMin{BOX_MIN} , boxMax{BOX_MAX};
            for(auto& corner : aabb){
                glm::vec3 worldCorner = glm::vec3(matrix * glm::vec4(corner , 1.0f));
                boxMin.x = std::min(boxMin.x , worldCorner.x);
                boxMin.y = std::min(boxMin.y , worldCorner.y);
                boxMin.z = std::min(boxMin.z , worldCorner.z);

                boxMax.x = std::max(boxMax.x , worldCorner.x);
                boxMax.y = std::max(boxMax.y , worldCorner.y);
                boxMax.z = std::max(boxMax.z , worldCorner.z);
            }

            const glm::vec3 center = (boxMin + boxMax) * 0.5f; 
            const float radius = glm::length(boxMax - center);

            {
                MeshData data{};
                data = baseMeshData;
                data.boxCenter = center;
                data.boxMax = boxMax;
                data.boxMin = boxMin;
                data.boxRadius = radius;
                data.instanceIndex = instanceID;
                totalMeshesData.push_back(data);
            }

            {
                InstanceData data{};
                data.meshDataIndex = meshID;
                data.LUT_index = LUT_index;
                data.BLAS_index=  BLAS_index;
                data.needsToBeDraw = true;

                instanceLayoutIndices.push_back(instanceInfoID);
                totalInstanceLayouts.push_back(data);
            }

            needsUpdate = true;
        }

    public:

        void updateMatrix(const glm::mat4 matrix , const uint32_t index = 0 ) noexcept{
            std::lock_guard<std::mutex> guard(m);
            if(index >= instanceLayoutIndices.size()){
                static uint32_t totalMessageCount = 0;
                if(totalMessageCount < 10){
                    std::cerr << std::format("couldnt update matrix for mesh , since the matrix index is out of range, message count -> {}", totalMessageCount ) << '\n';
                    totalMessageCount++;
                }
                return;
            }

            auto aabb = GetAABBCorners(localBoxMin , localBoxMax);
            glm::vec3 boxMin{BOX_MIN} , boxMax{BOX_MAX};
            for(auto& corner : aabb){
                glm::vec3 worldCorner = glm::vec3(matrix * glm::vec4(corner , 1.0f));
                boxMin.x = std::min(boxMin.x , worldCorner.x);
                boxMin.y = std::min(boxMin.y , worldCorner.y);
                boxMin.z = std::min(boxMin.z , worldCorner.z);

                boxMax.x = std::max(boxMax.x , worldCorner.x);
                boxMax.y = std::max(boxMax.y , worldCorner.y);
                boxMax.z = std::max(boxMax.z , worldCorner.z);
            }

            const glm::vec3 center = (boxMin + boxMax) * 0.5f; 
            const float radius = glm::length(boxMax - center);

            const uint32_t instanceInfoID = instanceLayoutIndices[index];
            const uint32_t meshID = totalInstanceLayouts[instanceInfoID].meshDataIndex;
            totalMeshesData[meshID].boxMax = boxMax;
            totalMeshesData[meshID].boxMin = boxMin;
            totalMeshesData[meshID].boxRadius = radius;
            totalMeshesData[meshID].boxCenter = center;
            const uint32_t instanceID = totalMeshesData[meshID].instanceIndex;
            instances[instanceID] = matrix;

            needsUpdate = true;
        }

        void deleteMatrix(const uint32_t index) noexcept{
            std::lock_guard<std::mutex> guard(m);
            if(index >= instanceLayoutIndices.size()){
                static uint32_t totalMessageCount = 0;
                if(totalMessageCount < 10){
                    std::cerr << std::format("couldnt delete matrix for mesh , since the matrix index is out of range, message count -> {}", totalMessageCount ) << '\n';
                    totalMessageCount++;
                }
                return;
            }

            const uint32_t instanceInfoID = instanceLayoutIndices[index];

            totalInstanceLayouts[instanceInfoID].needsToBeDraw = false;

            needsUpdate = true;
        }

        void changeLightStatus(const uint8_t status) noexcept{
            LUTs[LUT_index].isLightSource = status;
            needsUpdate = true;
        }

    private:

        MeshData baseMeshData{};

        [[nodiscard]] ErrorDataOutput<void> processInit(CreateInfo& input) noexcept{

            baseMeshData.firstIndex = totalIndices.size();
            baseMeshData.indexCount = input.inputIndices.size();
            baseMeshData.firstVertex = totalVertices.size();

            if(input.needsCalcTBN){
                // the tangent and the direction of v are summed over every triangle that uses a vertex ( a shared vertex used to take
                // what the last triangle said ) and turned into the final frame afterwards
                std::vector<glm::vec3> tangentSum(input.inputVertices.size() , glm::vec3(0.0f));
                std::vector<glm::vec3> dPdvSum(input.inputVertices.size() , glm::vec3(0.0f));

                for(uint32_t i = 0 ; i < baseMeshData.indexCount ; i +=3){
                    const uint32_t ia = input.inputIndices[i];
                    const uint32_t ib = input.inputIndices[i + 1];
                    const uint32_t ic = input.inputIndices[i + 2];
                    if(ia >= input.inputVertices.size() || ib >= input.inputVertices.size() || ic >= input.inputVertices.size()) continue;

                    const Vertex& a = input.inputVertices[ia];
                    const Vertex& b = input.inputVertices[ib];
                    const Vertex& c = input.inputVertices[ic];

                    const glm::vec3 edge1 = b.position - a.position;
                    const glm::vec3 edge2 = c.position - a.position;

                    const glm::vec2 uv1 = b.UV - a.UV;
                    const glm::vec2 uv2 = c.UV - a.UV;

                    // the sign of the determinant is the handedness of the UVs ( a mirrored part has a negative one ) , it has to stay :
                    // the old std::max( det , 0.00001 ) turned every mirrored triangle into a huge f . a triangle without an area in the
                    // UV space has no direction at all , it adds nothing
                    const float det = uv1.x * uv2.y - uv1.y * uv2.x;
                    if(std::abs(det) < 1e-8f) continue;
                    const float f = 1.0f / det;

                    const glm::vec3 dPdu = f * (edge1 * uv2.y - edge2 * uv1.y);
                    const glm::vec3 dPdv = f * (edge2 * uv1.x - edge1 * uv2.x);

                    // the weight is the area of the triangle : a big triangle says more than a sliver
                    const float weight = glm::length(glm::cross(edge1 , edge2));
                    for(const uint32_t index : {ia , ib , ic}){
                        if(glm::length(dPdu) > 1e-12f) tangentSum[index] += glm::normalize(dPdu) * weight;
                        if(glm::length(dPdv) > 1e-12f) dPdvSum[index] += glm::normalize(dPdv) * weight;
                    }
                }

                for(uint32_t v = 0 ; v < input.inputVertices.size() ; v++){
                    Vertex& vertex = input.inputVertices[v];
                    const glm::vec3 N = glm::normalize(vertex.normal);

                    glm::vec3 T = tangentSum[v] - glm::dot(N , tangentSum[v]) * N;
                    if(glm::length(T) < 1e-6f){
                        // no usable UVs around this vertex : any direction across the normal is better than NaN
                        const glm::vec3 helper = (std::abs(N.y) < 0.99f) ? glm::vec3(0.0f , 1.0f , 0.0f) : glm::vec3(1.0f , 0.0f , 0.0f);
                        T = glm::cross(helper , N);
                    }
                    T = glm::normalize(T);

                    // cross( N , T ) is the bitangent of the engine ( v going down the picture ) , a mirrored UV set flips it
                    const glm::vec3 B = glm::cross(N , T);
                    const float handedness = (glm::dot(B , dPdvSum[v]) > 0.0f) ? -1.0f : 1.0f;

                    vertex.tangent = T;
                    vertex.bitangent = B * handedness;
                }
            }

            VectorCat(totalVertices , input.inputVertices);
            VectorCat(totalIndices , input.inputIndices);

            localBoxMax = BOX_MAX;
            localBoxMin =BOX_MIN;
            for(auto& vertex : input.inputVertices){

                localBoxMax.x = glm::max(localBoxMax.x , vertex.position.x);
                localBoxMax.y = glm::max(localBoxMax.y , vertex.position.y);
                localBoxMax.z = glm::max(localBoxMax.z , vertex.position.z);

                localBoxMin.x = glm::min(localBoxMin.x , vertex.position.x);
                localBoxMin.y = glm::min(localBoxMin.y , vertex.position.y);
                localBoxMin.z = glm::min(localBoxMin.z , vertex.position.z);
            }

            LUT_index = LUTs.size();

            LUT lut{};
            lut.color = input.color;
            lut.emission = input.emission;
            lut.isLightSource = input.isALightObject;

            lut.metallic = input.metallic;
            lut.roughness = input.roughness;

            lut.depthTextureID = NON_VALID_SLOT;
            lut.diffuseTextureID = NON_VALID_SLOT;
            lut.normalTextureID = NON_VALID_SLOT;
            lut.metallicTextureID = NON_VALID_SLOT;
            lut.roughnessTextureID = NON_VALID_SLOT;

            if(input.diffusePath != NonTexturePath){
                Texture2::CreateInfo info{};
                info.path = input.diffusePath;
                info.context = context;
                info.format = VK_FORMAT_R8G8B8A8_UNORM;

                const auto result = manager->addTexture(info);
                if(result.has_value()){
                    lut.diffuseTextureID = result.value();
                }else{
                   std::cerr << std::format("couldnt find the diffuse path -> {} , despite the miss, mesh will be created" , input.diffusePath) << '\n';
                }
            }

            if(input.normalPath != NonTexturePath){
                Texture2::CreateInfo info{};
                info.path = input.normalPath;
                info.context = context;
                info.format = VK_FORMAT_R8G8_UNORM;

                const auto result = manager->addTexture(info);
                if(result.has_value()){
                    lut.normalTextureID = result.value();
                }else{
                   std::cerr << std::format("couldnt find the normal path -> {} , despite the miss, mesh will be created" , input.normalPath) << '\n';
                }
            }

            if(input.depthPath != NonTexturePath){
                Texture2::CreateInfo info{};
                info.path = input.depthPath;
                info.context = context;
                info.format = VK_FORMAT_R32_SFLOAT;

                const auto result = manager->addTexture(info);
                if(result.has_value()){
                    lut.depthTextureID = result.value();
                }else{
                   std::cerr << std::format("couldnt find the depth path -> {} , despite the miss, mesh will be created" , input.depthPath) << '\n';
                }
            }

            if(input.roughnessPath != NonTexturePath){
                Texture2::CreateInfo info{};
                info.path = input.roughnessPath;
                info.context = context;
                info.format = VK_FORMAT_R32_SFLOAT;

                const auto result = manager->addTexture(info);
                if(result.has_value()){
                    lut.roughnessTextureID = result.value();
                }else{
                   std::cerr << std::format("couldnt find the roughness path -> {} , despite the miss, mesh will be created" , input.roughnessPath) << '\n';
                }
            }

            if(input.metallicPath != NonTexturePath){
                Texture2::CreateInfo info{};
                info.path = input.metallicPath;
                info.context = context;
                info.format = VK_FORMAT_R32_SFLOAT;

                const auto result = manager->addTexture(info);
                if(result.has_value()){
                    lut.metallicTextureID = result.value();
                }else{
                   std::cerr << std::format("couldnt find the metallic path -> {} , despite the miss, mesh will be created" , input.metallicPath) << '\n';
                }
            }

            LUTs.push_back(lut);

            // now the blas
            BLAS_index = AS_layouts.size();
            {
                BLAS2_layout layout{};
                layout.firstVertex = baseMeshData.firstVertex;
                layout.indexCount = baseMeshData.indexCount;
                layout.indexOffset = baseMeshData.firstIndex;
                layout.vertexCount = input.inputVertices.size();

                AS_layouts.push_back(layout);
            }
            totalMeshes++;
            // the lock is already held by the constructor
            addMatrixLocked(glm::mat4(1.0f));
            return {};  
        }

        uint32_t LUT_index;
        uint32_t BLAS_index;

        glm::vec3 localBoxMin;
        glm::vec3 localBoxMax;

        std::vector<uint32_t> instanceLayoutIndices{};

        inline static VulkanContext* context;
        inline static TextureManager2* manager;

        inline static uint8_t startedRecording{false};
        inline static uint8_t endRecording{false};
        inline static uint8_t needsUpdate{true};

        inline static uint32_t totalMeshes = 0;

        inline static std::mutex m;

        void processCopy(Mesh2&& input) noexcept{
            // the old check compared firstIndex : it is 0 for the very first mesh and for a mesh that was just made , so that move copied nothing
            if(this == &input) return;

            baseMeshData = input.baseMeshData;
            LUT_index = input.LUT_index;
            BLAS_index = input.BLAS_index;
            instanceLayoutIndices = std::move(input.instanceLayoutIndices);
            input.instanceLayoutIndices.clear();
            localBoxMax = input.localBoxMax;
            localBoxMin = input.localBoxMin;
            // totalMeshes is not touched : a move makes no new mesh and destroys none ( Mesh2 has no destructor )
        }

        const glm::vec3 BOX_MIN = glm::vec3(FLT_MAX);

        const glm::vec3 BOX_MAX = glm::vec3(-FLT_MAX);

        inline std::vector<glm::vec3> GetAABBCorners(glm::vec3 boxMin, glm::vec3 boxMax) {
            std::vector<glm::vec3> corners(8);

            corners[0] = glm::vec3(boxMin.x, boxMin.y, boxMin.z);
            corners[1] = glm::vec3(boxMax.x, boxMin.y, boxMin.z);
            corners[2] = glm::vec3(boxMin.x, boxMax.y, boxMin.z);
            corners[3] = glm::vec3(boxMax.x, boxMax.y, boxMin.z);
            corners[4] = glm::vec3(boxMin.x, boxMin.y, boxMax.z);
            corners[5] = glm::vec3(boxMax.x, boxMin.y, boxMax.z);
            corners[6] = glm::vec3(boxMin.x, boxMax.y, boxMax.z);
            corners[7] = glm::vec3(boxMax.x, boxMax.y, boxMax.z);

            return corners;
        }

        void processDelete() noexcept{
            // totalMeshes++ is the last thing processInit does , so a mesh that failed in the constructor was never counted
            if(totalMeshes == 0){
                vertexBuffer = nullptr;
                indicesBuffer = nullptr;
                indirectCmdBuffer = nullptr;
                instanceBuffer = nullptr;
                BLAS_storage.clear();
                instanceLayoutBuffer = nullptr;
                LUT_buffer = nullptr;
                MeshBuffer = nullptr;
                context = nullptr;
                manager = nullptr;
            }
        }
    };

    using pMesh2 = std::shared_ptr<Mesh2>;
}