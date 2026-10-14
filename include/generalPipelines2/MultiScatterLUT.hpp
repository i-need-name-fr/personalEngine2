#pragma once

#include "../ShaderPipe/ShaderPipe.hpp"
#include "../Sampler/Sampler.hpp"

namespace mox{

    class MultiScatterLUTPipe final{
    public:

        SetTypicalOperators(MultiScatterLUTPipe)
        SetTypicalTransferOperators(MultiScatterLUTPipe)

        struct CreateInfo{
            VulkanContext* context;
            std::string path;
            RenderImage2* transmission;
            RenderImage2* image;
        };

        MultiScatterLUTPipe(const CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeQueueManager | VkReqTypeLogicalDevice , info.context);
                if(!result.has_value()){
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            if(!info.image || !info.transmission){
                engineLogger({MOX_ERROR_TYPE_FAILED_CREATION , "failed to create multi scatter pipe , since the render image or the transmission image is nullptr"});
                THROW_MESSAGE;
            }

            // one image can not be sampled and written by the same dispatch
            if(info.image == info.transmission){
                engineLogger({MOX_ERROR_TYPE_FAILED_CREATION , "failed to create multi scatter pipe , since the result is the transmission image itself"});
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

        ~MultiScatterLUTPipe() noexcept {
            processDelete();
        }

    private:

        pShaderPipe pipe{nullptr};
        pSampler sampler{nullptr};

        [[nodiscard]] ErrorDataOutput<void> createPipe(const CreateInfo& input) noexcept{

            try{

                pipe = std::make_shared<ShaderPipe>(input.context);

                {
                    Sampler::CreateInfo info{};
                    info.context = input.context;
                    info.linearClampSampler = true;

                    sampler = std::make_shared<Sampler>(info);
                }

                {
                    std::vector<DescriptorLayoutData> data(3);
                    data[0] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = input.context;
                    info.data = data;
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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create MultiScatterLUT pipe"});
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

            VkDescriptorImageInfo transmission{};
            transmission.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            transmission.imageView = info.transmission->view;
            transmission.sampler = nullptr;

            VkDescriptorImageInfo samplerInfo{};
            samplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            samplerInfo.imageView = nullptr;
            samplerInfo.sampler = sampler->sampler;

            VkDescriptorImageInfo output{};
            output.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            output.imageView = info.image->view;
            output.sampler = nullptr;

            std::array<VkWriteDescriptorSet , 3> writes{};
            for(uint32_t i = 0; i < writes.size(); i++){
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].pNext = nullptr;
                writes[i].dstSet = nullptr;
                writes[i].dstBinding = i;
                writes[i].dstArrayElement = 0;
                writes[i].descriptorCount = 1;
                writes[i].pBufferInfo = nullptr;
                writes[i].pTexelBufferView = nullptr;
            }
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[0].pImageInfo = &transmission;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            writes[1].pImageInfo = &samplerInfo;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[2].pImageInfo = &output;

            // the size of the LUT , not of the screen
            const uint32_t x = (info.image->extent.width + 7) / 8;
            const uint32_t y = (info.image->extent.height + 7) / 8;

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            cmdPushDescriptorSetKHR(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , static_cast<uint32_t>(writes.size()) , writes.data());

            {
                // the transmission LUT was written by the other pipe , this one reads it and writes the result
                std::array<VkImageMemoryBarrier2 , 2> barriers;
                barriers[0] = info.transmission->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                barriers[1] = info.image->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

                VkDependencyInfo dependency{};
                dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dependency.bufferMemoryBarrierCount = 0;
                dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
                dependency.pMemoryBarriers = nullptr;
                dependency.pBufferMemoryBarriers = nullptr;
                dependency.pImageMemoryBarriers = barriers.data();
                dependency.memoryBarrierCount = 0;
                dependency.pNext = nullptr;
                dependency.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &dependency);
            }

            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
        }

        void processCopy(MultiScatterLUTPipe&& input) noexcept{
            pipe = std::move(input.pipe);
            sampler = std::move(input.sampler);
        }

        void processDelete() noexcept{
            pipe = nullptr;
            sampler = nullptr;
        }
    };

    using pMultiScatterLUTPipe = std::shared_ptr<MultiScatterLUTPipe>;
}
