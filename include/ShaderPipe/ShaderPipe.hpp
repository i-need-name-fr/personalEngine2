#pragma once

#include "ShaderObject.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include <stdfloat>

namespace mox{

    class ShaderPipe{
    public:

        ShaderPipe() = delete;
        ShaderPipe(ShaderPipe& input) = delete;
        ShaderPipe& operator=(ShaderPipe& input) = delete;
        ShaderPipe& operator=(ShaderPipe&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }

        ShaderPipe(ShaderPipe&& input) noexcept{
            performCopy(std::move(input));
        }

        std::vector<pDescriptorLayout> descriptorLayouts{};
        std::vector<pShaderObject> shaders{};
        std::vector<pDescriptorBuffer2> buffers{};

        VkPipelineLayout layout = nullptr;

        std::vector<VkDescriptorSetLayout> getDescriptorLayoutData() const noexcept{
            if(descriptorLayouts.empty()) return {};
            std::vector<VkDescriptorSetLayout> data{};
            for(auto& l : descriptorLayouts){
                data.push_back(l->layout);
            }
            return data;
        }

        ShaderPipe(VulkanContext* context){
            if(!context){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shaderPipe , since the context is nullptr");
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(!context->logicalDevice){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shaderPipe , since the context logical device is nullptr");
                engineLogger(&error);
                THROW_MESSAGE;
            }
            
            device = context->logicalDevice->device;
        }

        ~ShaderPipe(){
            cleanPipe();
        }

    private:

        void cleanPipe() noexcept {
            if(device){
                if(layout) vkDestroyPipelineLayout(device , layout , nullptr);
            }

            layout = nullptr;

            descriptorLayouts.clear();
            shaders.clear();
            buffers.clear();
            device = nullptr;
        }

        void performCopy(ShaderPipe&& input) noexcept{
            // the pipeline layout and the shaders that this object held would stay alive for ever , and a move into itself emptied the vectors
            if(this == &input) return;
            cleanPipe();
            descriptorLayouts = std::move(input.descriptorLayouts);
            shaders = std::move(input.shaders);
            buffers = std::move(input.buffers);
            layout = std::exchange(input.layout , nullptr);
            device = std::exchange(input.device , nullptr);
        }

        VkDevice device = nullptr;

    };

    using pShaderPipe = std::shared_ptr<ShaderPipe>;
}
