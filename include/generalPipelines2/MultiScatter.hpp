#pragma once

#include "../Image2/RenderImage2.hpp"
#include "../ShaderPipe/ShaderPipe.hpp"

namespace mox{

    class MultiScatter final{
    public:

        SetTypicalOperators(MultiScatter);
        SetTypicalTransferOperators(MultiScatter)

        struct CreateInfo{
            VulkanContext* context;
            RenderImage2* image;
            std::string path;
        };

        explicit MultiScatter(CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager , info.context);
                if(!result.has_value()){
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }

            {
                const auto result = createPipe(info);
                if(!result.has_value()){
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }

            {
                const auto result = processTexture(info);
                if(!result.has_value()){
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }
        }

    private:    

        pShaderPipe pipe{nullptr};

        [[nodiscard]] ErrorDataOutput<void> processTexture(CreateInfo &input) noexcept{
            cmdFunctionType function = [&](VkCommandBuffer &cmd) -> void{
                executePass(cmd , input.image);
            };

            input.context->queueManager->Begin();
            input.context->queueManager->recordCmd(function , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
            {
                const auto result = input.context->queueManager->End();
                if(!result.has_value()) return result;
            }
            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe(CreateInfo &input) noexcept{
            try{
                pipe = std::make_shared<ShaderPipe>(input.context);

                {
                    DescriptorLayoutData data{};
                    data = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    DescriptorLayout::CreateInfo info{};
                    info.data = {data};
                    info.context= input.context;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.pNext = nullptr;

                    pipe->addDescriptorLayout(info);
                }

                {
                    PipelineLayout::CreateInfo info{};
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.context = input.context;
                    info.ranges = {};
                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {
                    DescriptorData2 data = DescriptorData2::createData(input.image , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0 , 0 );
                    DescriptorBuffer2::CreateInfo info{};
                    info.context = input.context;
                    info.data = {data};
                    info.layout = {pipe->descriptorLayouts[0].get()};

                    pipe->addDescriptorBuffer2(info);
                }

                {
                    ShaderObject::CreateInfo info{};
                    info.context = input.context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = input.path;
                    info.pushes = {};
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    pipe->addShader(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create MultiScatter pipe"});
            }
            return {};
        }

        void executePass(VkCommandBuffer& cmd , RenderImage2* image) noexcept {

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

            {
                const VkImageMemoryBarrier2 barrier = image->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 1;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = &barrier;
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);
            }

            vkCmdDispatch(cmd , (512 + 63) / 64 ,1, 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            {
                const VkImageMemoryBarrier2 barrier = image->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 1;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = &barrier;
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);
            }
        }

        void processCopy(MultiScatter&& input){
            pipe = std::move(input.pipe);
        }

        void processDelete(){
            pipe = nullptr;
        }
    };

    using pMultiScatter = std::shared_ptr<MultiScatter>;
}