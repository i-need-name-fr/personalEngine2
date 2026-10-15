#pragma once

#include "../ShaderPipe/ShaderPipe.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"

namespace mox{

    class ForwardPlus : public GraphPass2{
    public:

        SetTypicalOperators(ForwardPlus)
        SetTypicalTransferOperators(ForwardPlus)

        struct CreateInfo{
            VulkanContext* context;
            std::string path;
            ForwardPlusInfo generalData{};
        };

        explicit ForwardPlus(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~ForwardPlus(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(5);

            reqs[0].type = moxGraphTypeBuffer;
            reqs[0].name = "generalBuffer";

            reqs[1].type = moxGraphTypeBuffer;
            reqs[1].name = "forwardInfo";

            reqs[2].type = moxGraphTypeBuffer;
            reqs[2].name = "lightBuffer";

            reqs[3].type = moxGraphTypeBuffer;
            reqs[3].name = "tileLightIndices";

            reqs[4].type = moxGraphTypeBuffer;
            reqs[4].name = "lightCountPerTile";

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [ForwardPlus]"});

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.address = pipe->buffers[0]->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 1,  &info);

                uint32_t sets[1] = {0};
                VkDeviceSize offset[] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offset);
            }

            vkCmdDispatch(cmd , (inputData.generalData.TILE_TOTAL_AMOUNT + 256 - 1) / 256 , 1 ,1 );

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{

            return true;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindSources(sources ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer} , GraphSlot<BufferBase2>{"forwardInfo" , &forwardInfo} ,
                GraphSlot<BufferBase2>{"lightBuffer" , &lightBuffer} , GraphSlot<BufferBase2>{"tileLightIndices" , &tileLightIndices} ,
                GraphSlot<BufferBase2>{"lightCountPerTile" , &lightCountPerTile});
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [ForwardPlus]"});
            // lets clean 2 buffers
            {
                std::array<VkBufferMemoryBarrier2 , 2> barriers;
                barriers[0] = tileLightIndices->translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
                barriers[1] = lightCountPerTile->translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 5;
                info.imageMemoryBarrierCount = 0;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = barriers.data();
                info.pImageMemoryBarriers = nullptr;
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);

                vkCmdFillBuffer(cmd , tileLightIndices->buffer , 0 , tileLightIndices->getBufferSize() , 0);
                vkCmdFillBuffer(cmd , lightCountPerTile->buffer , 0 , lightCountPerTile->getBufferSize() , 0);

            }

            std::array<VkBufferMemoryBarrier2 , 5> barriers;

            barriers[0] = generalBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            barriers[1] = forwardInfo->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            barriers[2] = lightBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            barriers[3] = tileLightIndices->translateBuffer(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            barriers[4] = lightCountPerTile->translateBuffer(VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 5;
            info.imageMemoryBarrierCount = 0;
            info.pMemoryBarriers = nullptr;
            info.pBufferMemoryBarriers = barriers.data();
            info.pImageMemoryBarriers = nullptr;
            info.memoryBarrierCount = 0;
            info.pNext = nullptr;
            info.dependencyFlags = 0;

            vkCmdPipelineBarrier2(cmd , &info);

            return {};
        }

    private:

        pShaderPipe pipe{nullptr};
        CreateInfo inputData{};

        BufferBase2* generalBuffer{nullptr};
        BufferBase2* forwardInfo{nullptr};
        BufferBase2* lightBuffer{nullptr};
        BufferBase2* tileLightIndices{nullptr};
        BufferBase2* lightCountPerTile{nullptr};

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            VulkanContext* context = inputData.context;

            try{

                pipe = std::make_shared<ShaderPipe>(inputData.context);

                {
                    std::vector<DescriptorLayoutData> data(5);
                    data[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};


                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                    pipe->addDescriptorLayout(info);
                }

                {
                    PipelineLayout::CreateInfo info{};
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.context = context;
                    info.ranges = {};
                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {
                    std::vector<DescriptorData2> data(5);
                    data[0] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                    data[1] = DescriptorData2::createData(forwardInfo , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0);
                    data[2] = DescriptorData2::createData(lightBuffer , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 2 , 0);
                    data[3] = DescriptorData2::createData(tileLightIndices , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0);
                    data[4] = DescriptorData2::createData(lightCountPerTile , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->addDescriptorBuffer2(info);
                }

                {
                    ShaderObject::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.pushes = {};
                    info.filePath = inputData.path;
                    info.layouts = {pipe->descriptorLayouts[0].get()};

                    pipe->addShader(info);
                }

            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create FowardPlus pipe"});
            }
            completedInit = true;
            return {};
        }

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(ForwardPlus&& input) noexcept{
            if(this == &input) return;
            processDelete();

            pipe = std::move(input.pipe);
            inputData = input.inputData;

            // the buffers of the graph
            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            forwardInfo = std::exchange(input.forwardInfo , nullptr);
            lightBuffer = std::exchange(input.lightBuffer , nullptr);
            tileLightIndices = std::exchange(input.tileLightIndices , nullptr);
            lightCountPerTile = std::exchange(input.lightCountPerTile , nullptr);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete(void) noexcept{
            // the buffers belong to the graph , they are only forgotten
            pipe = nullptr;

            generalBuffer = nullptr;
            forwardInfo = nullptr;
            lightBuffer = nullptr;
            tileLightIndices = nullptr;
            lightCountPerTile = nullptr;

            inputData = {};
            completedInit = false;
        }
    };

    using pForwardPlus = std::shared_ptr<ForwardPlus>;
}