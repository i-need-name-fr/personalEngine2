#pragma once

#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../Buffer2/GeneralBuffer.hpp"
#include "../ShaderPipe2/ShaderPipeSelect.hpp"

namespace mox{

    // the corner of the grid with the smallest x y z ( world space ) : the player snapped down to a whole cell , minus half of the cube
    // GRID_SIZE is the number of the cells on an axis , the player is in the middle of the cube : half of the cells ( in meters ) is on each side
    inline glm::vec3 getGridOrigin(const glm::vec3 playerPosition) noexcept{
        const glm::vec3 snapped = glm::floor(playerPosition / lg::GRID_CELL_SIZE) * lg::GRID_CELL_SIZE;
        return snapped - glm::vec3(lg::GRID_SIZE * 0.5f * lg::GRID_CELL_SIZE);
    }

    inline glm::vec3 translateToGridPos(const glm::vec3 position , const glm::vec3 playerPosition) noexcept{
        return position - getGridOrigin(playerPosition);
    }

    inline int32_t translatePosToCellIndex(const glm::vec3 gridPos) noexcept{
        const glm::vec3 cell = glm::floor(gridPos / lg::GRID_CELL_SIZE);
        const float n = lg::GRID_SIZE;
        if(cell.x < 0.0f || cell.y < 0.0f || cell.z < 0.0f || cell.x >= n || cell.y >= n || cell.z >= n) return NON_VALID_SLOT;

        const int32_t count = static_cast<int32_t>(lg::GRID_SIZE);
        return static_cast<int32_t>(cell.x) + count * (static_cast<int32_t>(cell.y) + count * static_cast<int32_t>(cell.z));
    }

    /*

    [[vk::binding(0 , 0)]] lightBuffer
    [[vk::binding(1 , 0)]] cellBuffer
    [[vk::binding(2 , 0)]] cascadeBuffer
    
    */

    class LightManager final{
    public:

        SetTypicalOperators(LightManager)
        SetTypicalTransferOperators(LightManager)

        struct CreateInfo{
            VulkanContext* context;
            Player* user;
            std::string shaderPath;
        };

        explicit LightManager(const CreateInfo &info)
        {
            {
                const auto result = checkRequirements(VkReqTypeQueueManager| VkReqTypeLogicalDevice | VkReqTypePhysicalDevice , info.context);
                if(!result.has_value())
                {
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            {
                if(!info.user)
                {
                    engineLogger({MOX_ERROR_TYPE_FAILED_CREATION , "failed to create light manager , since the user data is nullptr"});
                    THROW_MESSAGE;
                }
            }

            {
                if(info.shaderPath.empty())
                {
                    engineLogger({MOX_ERROR_TYPE_FAILED_CREATION , "failed to create light manager , since the path of the grid shader is empty"});
                    THROW_MESSAGE;
                }
            }

            context = info.context;
            user = info.user;
            {
                const auto result = _implInitBuffers(context);
                if(!result.has_value())
                {
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }

            {
                const auto result = _implCreatePipe(info.shaderPath);
                if(!result.has_value())
                {
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }
        }

        ~LightManager() noexcept{
            processDelete();
        }

        pDescriptorBuffer2 dbuffer;
        pDeviceReadableBuffer lightBuffer;
        pDeviceReadableBuffer lightCascadeBuffer;
        pDeviceReadableBuffer lightCellBuffer;
        pDescriptorLayout layout;

        [[gnu::hot]] void updateGrid(VkCommandBuffer& cmd) noexcept
        {
            _implUpdateGrid(cmd , user->position);
        }

        [[nodiscard]] ErrorDataOutput<void> updateGridAuto() noexcept
        {
            return _implUpdateGridAuto(user->position);
        }

        void addLight(const lg::LightSource light , const std::string name) noexcept
        {
            if(nameToPhysical.contains(name)) return;
            
            const int32_t slot = _implAddLight(light);
            [[unlikely]] if(slot == NON_VALID_SLOT)
            {
                std::cerr << "all slots for light has been taken , please delete one\n";
                return;
            }

            std::pair<std::string , std::int32_t> data{name , slot};
            nameToPhysical.insert(data);
        }

        void deleteLight(const std::string lightName) noexcept
        {
            [[unlikely]] if(!nameToPhysical.contains(lightName)) return;
            const uint32_t lightIndex = nameToPhysical[lightName];
            _implDeleteLightSource(lightIndex);
            nameToPhysical.erase(lightName);
        }

        void changeLight(const std::string _lightName , const lg::LightSource _newLight) noexcept{
            if(!nameToPhysical.contains(_lightName)) return;
            const int32_t id = nameToPhysical[_lightName];
            _implChangeLightSource(_newLight , id);
        }

    private:

        void processCopy(LightManager&& input) noexcept
        {
            dbuffer = std::move(input.dbuffer);
            lightBuffer = std::move(input.lightBuffer);
            lightCascadeBuffer = std::move(input.lightCascadeBuffer);
            lightCellBuffer = std::move(input.lightCellBuffer);
            layout = std::move(input.layout);
            pipe = std::move(input.pipe);

            context = std::exchange(input.context , nullptr);
            user = std::exchange(input.user , nullptr);
            cascadeInfo = input.cascadeInfo;
            allLights = input.allLights;
            physicalSlots = input.physicalSlots;
            nameToPhysical = std::move(input.nameToPhysical);
            totalLightCount = std::exchange(input.totalLightCount , 0u);
        }

        void processDelete() noexcept
        {
            dbuffer = nullptr;
            lightBuffer = nullptr;
            lightCascadeBuffer = nullptr;
            lightCellBuffer = nullptr;
            layout = nullptr;
            pipe = nullptr;

            context = nullptr;
            user = nullptr;
            nameToPhysical.clear();
            physicalSlots.fill(0);
            totalLightCount = 0;
        }

        void _implUpdateGrid(VkCommandBuffer& cmd , const glm::vec3 playerPos) noexcept
        {
            cascadeInfo.totalLightCount = totalLightCount;
            cascadeInfo.totalLightCells = TOTAL_CELLS;
            lightBuffer->updateBufferCmd(cmd , translateToSpan(allLights.data() , allLights.size()) , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            lightCascadeBuffer->updateBufferCmd(cmd , translateToSpan(&cascadeInfo , 1) , VK_ACCESS_2_UNIFORM_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            const VkAccessFlags2 cellAccess = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            const VkPipelineStageFlags2 cellStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

            // translateBuffer gives the barrier from the state that the buffer was in last time to the new one
            auto cellBarrier = [&]() -> void{
                VkBufferMemoryBarrier2 barrier = lightCellBuffer->translateBuffer(cellAccess , cellStage);

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.pNext = nullptr;
                info.dependencyFlags = 0;
                info.memoryBarrierCount = 0;
                info.pMemoryBarriers = nullptr;
                info.bufferMemoryBarrierCount = 1;
                info.pBufferMemoryBarriers = &barrier;
                info.imageMemoryBarrierCount = 0;
                info.pImageMemoryBarriers = nullptr;

                vkCmdPipelineBarrier2(cmd , &info);
            };

            cellBarrier();

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.address = dbuffer->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 1 , &info);

                const uint32_t sets[1] = {0};
                const VkDeviceSize offsets[1] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offsets);
            }

            GridPush push{};
            push.origin = getGridOrigin(playerPos);

            constexpr uint32_t groupSize = 256;

            push.mode = 0;
            vkCmdPushConstants(cmd , pipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(GridPush) , &push);
            vkCmdDispatch(cmd , (TOTAL_CELLS + groupSize - 1) / groupSize , 1 , 1);

            cellBarrier();

            push.mode = 1;
            vkCmdPushConstants(cmd , pipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(GridPush) , &push);
            vkCmdDispatch(cmd , (lg::MAX_LIGHT_SOURCES + groupSize - 1) / groupSize , 1 , 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
        }

        [[nodiscard]] ErrorDataOutput<void> _implUpdateGridAuto(const glm::vec3 playerPos) noexcept
        {

            context->queueManager->Begin();

            cmdFunctionType function = [&](VkCommandBuffer& cmd) -> void{
                _implUpdateGrid(cmd , user->position);
            };

            context->queueManager->recordCmd(function  , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

            {
                const auto result = context->queueManager->End();
                if(!result.has_value()) return result;
            }

            return {};
        }

        int32_t _implAddLight(const lg::LightSource light) noexcept
        {

            int32_t freePhysicalSlot = NON_VALID_SLOT;
            for(uint32_t i = 0 ; i < lg::MAX_LIGHT_SOURCES ; i++){
                if(!physicalSlots[i])
                {
                    freePhysicalSlot = i;
                    break;
                }
            }
            // if we didnt manage to find a free slot , we just send the error free slot index , since the public function will analyze it 
            // and will output error like couldnt add the light source , since 
            if(freePhysicalSlot == NON_VALID_SLOT) return freePhysicalSlot;
            physicalSlots[freePhysicalSlot] = 1;
            allLights[freePhysicalSlot] = light;
            totalLightCount++;
            return freePhysicalSlot;
        }

        void _implDeleteLightSource(const uint32_t lightIndex) noexcept{
            if(totalLightCount == 0) return;
            if(lightIndex >= lg::MAX_LIGHT_SOURCES) return;
            const lg::LightSource light = allLights[lightIndex];
            allLights[lightIndex] = {};
            physicalSlots[lightIndex] = 0;
            totalLightCount--;
        }

        void _implChangeLightSource(const lg::LightSource changedLight , const uint32_t lightIndex) noexcept{
            if(lightIndex >= lg::MAX_LIGHT_SOURCES)
            {
                static uint32_t totalMessageCount = 10;
                if(totalMessageCount > 0){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to rewrite existing light source , index -> {} , since its out of range" , lightIndex)};
                    engineLogger(error);
                    totalMessageCount--;
                }
                return;
            }
            allLights[lightIndex] = changedLight;
        }

        // the compute pipe of the grid : the set 0 is the layout and the descriptor buffer of the manager ( the lamps , the cells , the info ) , one push constant
        [[nodiscard]] ErrorDataOutput<void> _implCreatePipe(const std::string shaderPath) noexcept{
            try{
                pipe = std::make_shared<ShaderPipeType>(context);

                VkPushConstantRange range{};
                range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                range.offset = 0;
                range.size = sizeof(GridPush);

                {
                    PipelineLayout::CreateInfo info{};
                    info.context = context;
                    info.layouts = {layout.get()};
                    info.ranges = {range};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.pushes = {range};
                    info.filePath = shaderPath;
                    info.layouts = {layout.get()};

                    pipe->addShader(info);
                }
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the pipe of LightManager ( the grid shader )"});
            }
            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> _implInitBuffers(VulkanContext* context) noexcept{
            try{

                {
                    // first the descriptor layout
                    std::vector<DescriptorLayoutData> data(3);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    layout = std::make_shared<DescriptorLayout>(info);
                }

                // now all buffers
                // light buffer
                {
                    DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                    info.size = sizeof(lg::LightSource) * lg::MAX_LIGHT_SOURCES;

                    lightBuffer = std::make_shared<DeviceReadableBuffer>(info);
                }

                {
                    // light Cell info
                    DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                    info.size = sizeof(lg::LightCell) * TOTAL_CELLS;

                    lightCellBuffer = std::make_shared<DeviceReadableBuffer>(info);
                }

                // constant buffer
                {
                    DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                    info.size = sizeof(lg::GridCascadeInfo);

                    lightCascadeBuffer = std::make_shared<DeviceReadableBuffer>(info);
                }

                // descriptor buffer
                {
                    std::vector<DescriptorData2> data(3);
                    data[0] = DescriptorData2::createData(lightBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0);
                    data[1] = DescriptorData2::createData(lightCellBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0);
                    data[2] = DescriptorData2::createData(lightCascadeBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = layout.get();

                    dbuffer = std::make_shared<DescriptorBuffer2>(info);
                }
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create LightManager for new world grid type "});
            }

            return {};
        }

        VulkanContext* context;
        // the cells are only on the GPU ( the shader of the grid makes them ) : the number of them is GRID_SIZE ^ 3
        static constexpr uint32_t TOTAL_CELLS = static_cast<uint32_t>(lg::GRID_SIZE) * static_cast<uint32_t>(lg::GRID_SIZE) * static_cast<uint32_t>(lg::GRID_SIZE);

        // the push constant of LightGridUpdate.hlsl : the float3 and the uint share the same 16 bytes
        struct GridPush{
            alignas(16) glm::vec3 origin;
            uint32_t mode;
        };
        static_assert(sizeof(GridPush) == 16);

        pShaderPipeType pipe{nullptr};
        lg::GridCascadeInfo cascadeInfo;
        std::array<lg::LightSource , lg::MAX_LIGHT_SOURCES> allLights{};
        std::array<std::int32_t , lg::MAX_LIGHT_SOURCES> physicalSlots{};
        std::unordered_map<std::string , std::int32_t> nameToPhysical{};
        uint32_t totalLightCount = 0;
        Player* user;
    };

    using pLightManager = std::shared_ptr<LightManager>;
}