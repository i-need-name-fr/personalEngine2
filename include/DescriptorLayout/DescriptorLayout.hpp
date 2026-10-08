#pragma once

#include "../vulkanContext/vulkanBase.hpp"

namespace mox{

    struct DescriptorLayoutData{
        VkDescriptorType type{};
        uint32_t count = 1;
    };


    class DescriptorLayout final{
    public:

        DescriptorLayout() = delete;
        DescriptorLayout(DescriptorLayout& input) = delete;
        DescriptorLayout(const DescriptorLayout& input) = delete;
        DescriptorLayout& operator=(DescriptorLayout& input) = delete;
        DescriptorLayout& operator=(const DescriptorLayout& input) = delete;

        DescriptorLayout(DescriptorLayout&& input) noexcept{
            if(layout == input.layout) return;
            processCopy(std::move(input));
        }
        DescriptorLayout& operator=(DescriptorLayout&& input) noexcept{
            if(layout == input.layout) return *this;
            processCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context = nullptr;
            std::vector<DescriptorLayoutData> data{};
            VkDescriptorSetLayoutCreateFlags flags{0};
            VkShaderStageFlags stage{VK_SHADER_UNUSED_KHR};
            void* pNext = nullptr;
        };

        explicit DescriptorLayout(const CreateInfo& info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice , info.context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }

            device = info.context->logicalDevice->device;
            const auto result = createLayout(info);
            if(!result.has_value()){
                engineLogger(result.error());
                processDelete();
                THROW_MESSAGE;
            }
        }

        ~DescriptorLayout(){
            processDelete();
        }

        VkDescriptorSetLayout layout{nullptr};

    private:

        VkDevice device = nullptr;

        [[nodiscard]] ErrorDataOutput<void> createLayout(const CreateInfo &input) noexcept{

            std::vector<VkDescriptorSetLayoutBinding> bind{};
            {
                uint32_t i = 0 ;
                for(auto& dLayoutData : input.data){
                    VkDescriptorSetLayoutBinding binding{};
                    binding.binding = i;
                    binding.descriptorCount = dLayoutData.count;
                    binding.descriptorType = dLayoutData.type;
                    binding.pImmutableSamplers = nullptr;
                    binding.stageFlags = input.stage;

                    bind.push_back(std::move(binding));
                    i++;
                }
            }

            VkDescriptorSetLayoutCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            info.bindingCount = static_cast<uint32_t>(bind.size());
            info.pBindings = bind.data();
            info.flags = input.flags;
            info.pNext = input.pNext;
            
            const auto result = vkCreateDescriptorSetLayout(input.context->logicalDevice->device , &info , nullptr, &layout);
            if(result != VK_SUCCESS) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create descriptor set layout , error ID -> {}" , static_cast<int32_t>(result))});

            return {};
        }

        void processCopy(DescriptorLayout&& input) noexcept{
            processDelete();
            device = std::exchange(input.device , nullptr);
            layout = std::exchange(input.layout , nullptr);
        }

        void processDelete() noexcept{
            if(device && layout) vkDestroyDescriptorSetLayout(device , layout , nullptr);
            device = nullptr;
            layout = nullptr;
        }
    };

    using pDescriptorLayout = std::shared_ptr<DescriptorLayout>;
}