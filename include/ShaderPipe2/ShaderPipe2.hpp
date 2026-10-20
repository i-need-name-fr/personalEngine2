#pragma once

#include "ShaderObject2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../PipelineLayout/PipelineLayout.hpp"

namespace mox{

    class ShaderPipe2{
    public:

        SetTypicalOperators(ShaderPipe2)
        SetTypicalTransferOperators(ShaderPipe2)

        std::vector<pDescriptorLayout> descriptorLayouts{};
        std::vector<pShaderObject2> shaders{};
        std::vector<pDescriptorBuffer2> buffers{};

        pPipelineLayout layout = nullptr;

        std::vector<VkDescriptorSetLayout> getDescriptorLayoutData() const noexcept{
            if(descriptorLayouts.empty()) return {};
            std::vector<VkDescriptorSetLayout> data{};
            for(auto& l : descriptorLayouts){
                data.push_back(l->layout);
            }
            return data;
        }

        ShaderPipe2(VulkanContext* info){
            if(!info){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shaderPipe , since the context is nullptr");
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(!info->logicalDevice){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shaderPipe , since the context logical device is nullptr");
                engineLogger(&error);
                THROW_MESSAGE;
            }
            
            device = info->logicalDevice->device;
            this->context = info;
        }

        void addDescriptorBuffer2(const DescriptorBuffer2::CreateInfo &info){
            try{
                buffers.push_back(std::move(std::make_shared<DescriptorBuffer2>(info)));
            }
            catch(std::exception &e){throw std::runtime_error("failed to create descriptor buffer");}
        }

        void addDescriptorLayout(const DescriptorLayout::CreateInfo &info){
            try{
                descriptorLayouts.push_back(std::move(std::make_shared<DescriptorLayout>(info)));
            }
            catch(std::exception &e){throw std::runtime_error("failed to create descriptor layout");}
        }

        void addShader(const ShaderObject2::CreateInfo &info){
            try{
                shaders.push_back(std::move(std::make_shared<ShaderObject2>(info)));
            }
            catch(std::exception &e){throw std::runtime_error("failed to create shader");}
        }

        [[nodiscard]] ErrorDataOutput<void> createLayouts(std::vector<DescriptorLayout::CreateInfo> layouts , std::vector<VkPushConstantRange> ranges , VkShaderStageFlags stage) noexcept{
            try{

                for(auto& l :layouts){
                    addDescriptorLayout(l);
                }

                std::vector<DescriptorLayout*> data;

                for(auto& l : descriptorLayouts){
                    data.push_back(l.get());
                }

                PipelineLayout::CreateInfo info{};
                info.context = context;
                info.layouts = data;
                info.ranges = ranges;

                layout = std::make_shared<PipelineLayout>(info);
            }
            catch(std::exception&e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create layouts for shader pipe"});
            }
            return{};
        }

        ~ShaderPipe2(){
            cleanPipe();
        }

    private:

        void cleanPipe() noexcept {
            layout = nullptr;
            descriptorLayouts.clear();
            shaders.clear();
            buffers.clear();
            device = nullptr;
            context = nullptr;
        }

        void processCopy(ShaderPipe2&& input) noexcept{
            // the pipeline layout and the shaders that this object held would stay alive for ever , and a move into itself emptied the vectors
            if(this == &input) return;
            cleanPipe();
            descriptorLayouts = std::move(input.descriptorLayouts);
            shaders = std::move(input.shaders);
            buffers = std::move(input.buffers);
            layout = std::move(input.layout);
            device = std::exchange(input.device , nullptr);
            context = std::exchange(input.context, nullptr);
        }

        VkDevice device = nullptr;
        VulkanContext* context;
    };

    using pShaderPipe2 = std::shared_ptr<ShaderPipe2>;
}
