#pragma once

#include "../DescriptorLayout/DescriptorLayout.hpp"

namespace mox{

    class PipelineLayout final{
    public:

        PipelineLayout() = delete;
        PipelineLayout(PipelineLayout& input) = delete;
        PipelineLayout(const PipelineLayout& input) = delete;
        PipelineLayout& operator=(PipelineLayout& input) = delete;
        PipelineLayout& operator=(const PipelineLayout& input) = delete;

        PipelineLayout(PipelineLayout&& input) noexcept{
            if(this == &input) return;
            if(layout && context) vkDestroyPipelineLayout(context->logicalDevice->device , layout , nullptr);

            layout = std::exchange(input.layout , nullptr);
            context = std::exchange(input.context, nullptr);

        }
        PipelineLayout& operator=(PipelineLayout&& input) noexcept{
            if(this == &input) return *this;
            if(layout && context) vkDestroyPipelineLayout(context->logicalDevice->device , layout , nullptr);

            layout = std::exchange(input.layout , nullptr);
            context = std::exchange(input.context, nullptr);

            return *this;
        }

        ~PipelineLayout() noexcept{
            if(layout && context) vkDestroyPipelineLayout(context->logicalDevice->device , layout , nullptr);
            layout = nullptr;
            context = nullptr;
        }

        struct CreateInfo{
            VulkanContext* context;

            std::vector<DescriptorLayout*> layouts;
            std::vector<VkPushConstantRange> ranges;
        };

        PipelineLayout(const CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice , info.context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }

            std::vector<VkDescriptorSetLayout> layouts{};
            for(auto input : info.layouts){
                if(!input) continue;
                if(!input->layout)continue;
                layouts.push_back(input->layout);
            }

            if(layouts.size() != info.layouts.size()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create PipelineLayout , since the layouts contains some nullptr layout"});
                THROW_MESSAGE;
            }

            VkPipelineLayoutCreateInfo data{};
            data.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            data.setLayoutCount = static_cast<uint32_t>(layouts.size());
            data.pSetLayouts = layouts.data();
            data.pushConstantRangeCount = static_cast<uint32_t>(info.ranges.size());
            data.pPushConstantRanges = info.ranges.data();
            data.flags = 0;
            data.pNext = nullptr;
            context = info.context;

            const auto result = vkCreatePipelineLayout(context->logicalDevice->device , &data , nullptr , &layout);
            if(result != VK_SUCCESS){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create descriptor set layout , error ID -> {}" , static_cast<int32_t>(result))});
                if(layout ) vkDestroyPipelineLayout(context->logicalDevice->device , layout , nullptr);
                context = nullptr;
                THROW_MESSAGE;
            }

        }

        VkPipelineLayout layout{nullptr};

    private:

        VulkanContext* context{nullptr};
    };

    using pPipelineLayout = std::shared_ptr<PipelineLayout>;
}