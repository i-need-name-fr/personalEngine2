#pragma once

#include "../vulkanContext/vulkanBase.hpp"

namespace mox{

    class ShaderModule final{
    private:

        std::vector<char> readFile(std::string const path)noexcept{
            if(path.empty()) return {};
            std::ifstream file(path , std::ios::ate | std::ios::binary);
            if(!file.is_open()) return {};
            const uint64_t size  = file.tellg();
            file.seekg(0);
            std::vector<char> buffer(size);
            file.read(buffer.data() , size);
            file.close();
            return buffer;
        }

        ErrorDataOutput<void> createShaderModule(const std::string path) noexcept {
            auto buffer = readFile(path);
            if(buffer.empty()){
                ErrorDataType const error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader module of file - >  {}" , path)};
                return std::unexpected(error);
            }

            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = static_cast<uint32_t>(buffer.size());
            info.flags = 0;
            info.pNext = nullptr;
            info.pCode = std::bit_cast<const uint32_t*>(buffer.data());

            auto result = vkCreateShaderModule(device , &info , nullptr , &module);
            if(result != VK_SUCCESS){
                ErrorDataType const error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader module PATH -> {} : Error ID -> {}" , path , static_cast<int32_t>(result))};
                return std::unexpected(error);
            }

            return {};
        }

        VkDevice device = nullptr;

        void clean() noexcept {
            if(device){
                if(module) vkDestroyShaderModule(device , module , nullptr);
            }
        }

    public:

        VkShaderModule module = nullptr;

        ShaderModule() = delete;
        ShaderModule(ShaderModule& input) = delete;
        ShaderModule& operator=(ShaderModule& input) = delete;

        ShaderModule(ShaderModule&& input) noexcept {
            module = std::exchange(input.module , nullptr);
            device = std::exchange(input.device , nullptr);
        }

        ShaderModule& operator=(ShaderModule&& input) noexcept {
            module = std::exchange(input.module , nullptr);
            device = std::exchange(input.device , nullptr);

            return *this;
        }
        
        ShaderModule(VulkanContext* context , const std::string path){
            if(!context){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader module , since the context is nullptr");
                engineLogger(&error);
                THROW_MESSAGE;
            }
            if(!context->logicalDevice){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader module , since the logical device  is nullptr");
                engineLogger(&error);
                THROW_MESSAGE;
            }

            this->device = context->logicalDevice->device;
            auto result = createShaderModule(path);
            if(result.has_value()){

                ErrorDataType const success{MOX_ERROR_TYPE_SUCCESS , std::format("shader module -> {} has been created" , path)};

            }else{
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        ~ShaderModule(){clean();}
    };

    using pShaderModule = std::shared_ptr<ShaderModule>;

}