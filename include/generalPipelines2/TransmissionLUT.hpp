#pragma once

#include "../ShaderPipe/ShaderPipe.hpp"

namespace mox{

    class TransmissionLUTPipe final{
    public:

        SetTypicalOperators(TransmissionLUTPipe)
        SetTypicalTransferOperators(TransmissionLUTPipe)

        struct CreateInfo{
            VulkanContext* context;
            std::string path;
            RenderImage2* image;
        };

        TransmissionLUTPipe(const CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeQueueManager | VkReqTypeLogicalDevice , info.context);
                if(!result.has_value()){
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            if(!info.image){
                engineLogger({MOX_ERROR_TYPE_FAILED_CREATION , "failed to create transmission pipe , since the render image is nullptr"});
                THROW_MESSAGE;
            }

            {
                const auto result = createPipe(info);
                if(!result.has_value()){
                    processDelete();
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            {
                const auto result = executePipe(info);
                if(!result.has_value()){
                    processDelete();
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }
        }

        ~TransmissionLUTPipe() noexcept {
            processDelete();
        }

    private:

        pShaderPipe pipe{nullptr};

        [[nodiscard]] ErrorDataOutput<void> createPipe(const CreateInfo& input) noexcept{

            try{

                pipe = std::make_shared<ShaderPipe>(input.context);

                {
                    DescriptorLayoutData data{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = input.context;
                    info.data = {data};
                    info.pNext = nullptr;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                    pipe->addDescriptorLayout(info);
                }

                {
                    PipelineLayout::CreateInfo info{};
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.context= input.context;
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                // we dont need descriptor buffer

                {
                    ShaderObject::CreateInfo info{};
                    info.context = input.context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = input.path;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};

                    pipe->addShader(info);
                }

            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create TransmissionLUT pipe"});
            }

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> executePipe(const CreateInfo& info) noexcept{
            info.context->queueManager->Begin();

            cmdFunctionType function = [&](VkCommandBuffer &cmd) -> void{
                executionCmd(cmd , info);
            };

            info.context->queueManager->recordCmd(function , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            return info.context->queueManager->End();
        }

        void executionCmd(VkCommandBuffer &cmd , const CreateInfo &info) noexcept{

            VkDescriptorImageInfo image{};
            image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            image.imageView = info.image->view;
            image.sampler = nullptr;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pNext = nullptr;
            write.pImageInfo = &image;
            write.pBufferInfo = nullptr;
            write.pTexelBufferView = nullptr;
            write.dstArrayElement = 0;
            write.dstBinding = 0;
            write.dstSet = nullptr;

            // the size of the LUT , not of the screen
            const uint32_t x = (info.image->extent.width + 7) / 8;
            const uint32_t y = (info.image->extent.height + 7) / 8;

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            cmdPushDescriptorSetKHR(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1  , &write);

            {
                const auto barrier = info.image->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

                VkDependencyInfo dependency{};
                dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dependency.bufferMemoryBarrierCount = 0;
                dependency.imageMemoryBarrierCount = 1;
                dependency.pMemoryBarriers = nullptr;
                dependency.pBufferMemoryBarriers = nullptr;
                dependency.pImageMemoryBarriers = &barrier;
                dependency.memoryBarrierCount = 0;
                dependency.pNext = nullptr;
                dependency.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &dependency);
            }

            vkCmdDispatch(cmd , x , y ,1 );

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
        }

        void processCopy(TransmissionLUTPipe&& input) noexcept{
            pipe = std::move(input.pipe);
        }

        void processDelete() noexcept{
            pipe = nullptr;
        }
    };

    using pTransmissionLUTPipe = std::shared_ptr<TransmissionLUTPipe>;

    inline pTransmissionLUTPipe createTransmissionLUTPipe(const TransmissionLUTPipe::CreateInfo info){
        try{
            return std::make_shared<TransmissionLUTPipe>(info);
        }
        catch(std::exception& e) {throw;}
    }
}