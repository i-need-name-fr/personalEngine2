#pragma once

#include "../vulkanContext/vulkanBase.hpp"
#include <fstream>
#include <filesystem>
#include "../DescriptorLayout/DescriptorLayout.hpp"

namespace mox{

    class ShaderObject{
    public:

        ShaderObject() = delete;
        ShaderObject(ShaderObject& input) = delete;
        ShaderObject& operator=(ShaderObject& input) = delete;
        ShaderObject(ShaderObject&& input) noexcept{
            performCopy(std::move(input));
        }

        ShaderObject& operator=(ShaderObject&& input) noexcept{
            if(this == &input) return *this;
            // performCopy only takes the handle of the other shader , the shader that this object held would stay alive for ever
            if(device && shader) destroyShaderEXT(device , shader , nullptr);
            shader = nullptr;
            performCopy(std::move(input));

            return *this;
        }

        struct CreateInfo{
            VulkanContext* context;

            std::vector<DescriptorLayout*> layouts;
            std::vector<VkPushConstantRange> pushes;

            VkShaderStageFlagBits currentStage;
            VkShaderStageFlags nextStage;

            std::string filePath{};
        };

        ShaderObject(const CreateInfo &info){
            const auto result = createShader(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        ~ShaderObject(){
            if(device){
                if(shader) destroyShaderEXT(device , shader , nullptr);
            }

            shader=  nullptr;
            device = nullptr;
        }

        VkShaderEXT shader = nullptr;
    private:

        void performCopy(ShaderObject&& input) noexcept{
            shader = std::exchange(input.shader , nullptr);
            device = std::exchange(input.device,  nullptr);
        }

        [[nodiscard]] ErrorDataOutput<void> createShader(const CreateInfo &data) noexcept{

            if(!data.context){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the data context is nullptr"});
            }

            if(data.filePath.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the file path is nullptr"});
            }

            if(!data.context->logicalDevice){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the context logical device  is nullptr"});
            }

            device = data.context->logicalDevice->device;

            {
                std::error_code ec;
                std::filesystem::create_directories("../ShaderModuleCache" , ec);
            }

            std::vector<VkDescriptorSetLayout> layouts{};
            for(auto& layout : data.layouts){
                if(!layout) continue;
                layouts.push_back(layout->layout);
            }   

            if(layouts.size() != data.layouts.size()) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , due to some descriptor layouts being nullptr"});

            if(checkValidTemp(data.filePath)){

                auto buffer = getCacheData(data.filePath);
                if(buffer.empty()){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the cache buffer is nulltr"});
                }

                VkShaderCreateInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
                info.pNext = nullptr;
                info.flags = 0;
                info.pSpecializationInfo = nullptr;
                info.codeSize = static_cast<uint32_t>(buffer.size());
                info.codeType = VK_SHADER_CODE_TYPE_BINARY_EXT;
                info.nextStage = data.nextStage;
                info.pCode= buffer.data();
                info.stage = data.currentStage;
                info.setLayoutCount = static_cast<uint32_t>(data.layouts.size());

                info.pSetLayouts = layouts.data();
                info.pushConstantRangeCount = static_cast<uint32_t>(data.pushes.size());
                info.pPushConstantRanges = data.pushes.data();
                info.pName ="main";

                const auto result = createShaderEXT(device , 1 , &info , nullptr ,&shader);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader object, Error ID -> {}", static_cast<int32_t>(result))});
                }
                
            }else{

                auto buffer = getSPIR_V_code(data.filePath);
                if(buffer.empty()){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the cache buffer is nulltr"});
                }

                VkShaderCreateInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
                info.pNext = nullptr;
                info.flags = 0;
                info.pSpecializationInfo = nullptr;
                info.codeSize = static_cast<uint32_t>(buffer.size());
                info.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
                info.nextStage = data.nextStage;
                info.pCode= buffer.data();
                info.stage = data.currentStage;
                info.setLayoutCount = static_cast<uint32_t>(data.layouts.size());
                info.pSetLayouts = layouts.data();
                info.pushConstantRangeCount = static_cast<uint32_t>(data.pushes.size());
                info.pPushConstantRanges = data.pushes.data();
                info.pName ="main";

                const auto result = createShaderEXT(device , 1 , &info , nullptr ,&shader);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader object, Error ID -> {}", static_cast<int32_t>(result))});
                }

            }
            auto tempFilePath = getTempFileName(data.filePath);
            std::filesystem::copy(std::filesystem::path(data.filePath) , std::filesystem::path(tempFilePath) , std::filesystem::copy_options::overwrite_existing);

            saveToDisk(data.filePath);

            return {};
        }

        VkDevice device = nullptr;

        std::string getCacheFileName(const std::string& info)
        {
            const std::size_t directorySlot = info.find_last_of("/\\");
            const std::size_t nameStart = (directorySlot == std::string::npos) ? 0 : directorySlot + 1;   // +1, чтобы пропустить сам слэш; 0, если слэша нет

            const std::size_t dotSlot = info.find_last_of('.');
            const std::size_t nameLength = (dotSlot == std::string::npos) ? std::string::npos : dotSlot - nameStart;   // npos = "до конца", если точки нет

            std::string fileName = info.substr(nameStart, nameLength);

            return "../ShaderModuleCache/" + fileName + ".bin";
        }

        std::string getTempFileName(const std::string& info)
        {
            const std::size_t directorySlot = info.find_last_of("/\\");
            const std::size_t nameStart = (directorySlot == std::string::npos) ? 0 : directorySlot + 1;   // +1, чтобы пропустить сам слэш; 0, если слэша нет

            const std::size_t dotSlot = info.find_last_of('.');
            const std::size_t nameLength = (dotSlot == std::string::npos) ? std::string::npos : dotSlot - nameStart;   // npos = "до конца", если точки нет

            std::string fileName = info.substr(nameStart, nameLength);

            return "../ShaderModuleCache/" + fileName + "Temp.spv";
        }

        // returns false if there are changes
        bool checkValidTemp(const std::string file) noexcept{
            auto path = getTempFileName(file);

            if(!std::filesystem::exists(path)) return false;
            if (std::filesystem::file_size(file) != std::filesystem::file_size(path))
                return false;   

            std::ifstream a(file, std::ios::binary);   // std::ios::binary - ТА ЖЕ причина, что для SPIR-V/pipeline cache!
            std::ifstream b(path, std::ios::binary);

            return std::equal(
                std::istreambuf_iterator<char>(a),
                std::istreambuf_iterator<char>(),
                std::istreambuf_iterator<char>(b)
            );
        }

        std::vector<uint8_t> getCacheData(const std::string path) noexcept{
            auto file = getCacheFileName(path);
            if(!std::filesystem::exists(file)) return {};
            std::ifstream data(file , std::ios::in | std::ios::binary | std::ios::ate);
            if(data.is_open()){

                const std::size_t size = data.tellg();
                data.seekg(0);
                std::vector<uint8_t> buffer(size);
                data.read(std::bit_cast<char*>(buffer.data()) , size);
                data.close();
                
                return buffer;
            }
            return {};
        }

        std::vector<uint8_t> getSPIR_V_code(const std::string path) noexcept{
            if(!std::filesystem::exists(path)) return {};
            std::ifstream data(path , std::ios::in | std::ios::binary | std::ios::ate);
            if(data.is_open()){

                const std::size_t size = data.tellg();
                data.seekg(0);
                std::vector<uint8_t> buffer(size);
                data.read(std::bit_cast<char*>(buffer.data()) , size);
                data.close();
                
                return buffer;
            }
            return {};
        }

        void saveToDisk(const std::string &path){
            if(!shader) return;

            std::size_t count;
            getShaderBinaryDataEXT(device , shader , &count , nullptr);
            std::vector<uint8_t> buffer(count);
            getShaderBinaryDataEXT(device , shader , &count , buffer.data());

            auto pathToFile = getCacheFileName(path);
            std::ofstream file(pathToFile , std::ios::out | std::ios::binary);
            if(file.is_open()){
                file.write(std::bit_cast<const char*>(buffer.data()), count);
                file.close();
            }
        }
    };

    using pShaderObject = std::shared_ptr<ShaderObject>;
}


