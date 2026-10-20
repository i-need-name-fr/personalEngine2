#pragma once

#include "../vulkanContext/vulkanBase.hpp"
#include <fstream>
#include <filesystem>
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../RTCompileObjects/RTCompileObjects.hpp"

namespace mox{

    class ShaderObject2 final : public RTCompileObjects{
    public:

        ShaderObject2() = delete;
        ShaderObject2(ShaderObject2& input) = delete;
        ShaderObject2& operator=(ShaderObject2& input) = delete;
        ShaderObject2(ShaderObject2&& input) noexcept{
            performCopy(std::move(input));
        }

        ShaderObject2& operator=(ShaderObject2&& input) noexcept{
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

        ShaderObject2(const CreateInfo &info){
            const auto result = createShader(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        ~ShaderObject2(){
            if(device){
                if(shader) destroyShaderEXT(device , shader , nullptr);
            }

            shader=  nullptr;
            device = nullptr;
        }

        VkShaderEXT shader = nullptr;
    private:

        void performCopy(ShaderObject2&& input) noexcept{
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
            // the header is asked FIRST and always : with && it was skipped when the hlsl itself changed , the stamp of the header was then saved below ( commit )
            // before any other shader had compared with it , and those shaders took their old binaries
            const bool structureValid = checkValidShaderStructure();
            // no .spv -> it has to be compiled ( the .bin alone is not enough : the other mode of the build loads the .spv )
            std::string spvName{} , spvFolder{};
            parseFilePath(data.filePath , spvFolder , spvName);
            const bool spvExists = std::filesystem::exists(getCompiledShaderPath(spvName , spvFolder));
            const bool diffStatus = checkValidTemp(data.filePath) && structureValid && spvExists;
            if(diffStatus){

                auto buffer = getCacheData(data.filePath);
                if(buffer.empty()){
                    engineLogger(ErrorDataType{MOX_ERROR_TYPE_WARNING , "failed to create shader object , since the cache buffer is nulltr. however will not try to compile it within SPIR_V,"});
                    const auto result = compileShaderObject(data, layouts);
                    if(!result.has_value()) return result;
                    saveToDisk(data.filePath , false);
                    return {};
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
                    engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader object , however will not try to compile it within SPIR_V, Error ID -> {}", static_cast<int32_t>(result))});
                    const auto result = compileShaderObject(data, layouts);
                    if(!result.has_value()) return result;
                    saveToDisk(data.filePath , false);
                    return {};
                }
                
            }else{
                // first we compile the shader
                const auto result = compileShaderObject(data, layouts);
                if(!result.has_value()) return result;
            }
            auto tempFilePath = getTempFileName(data.filePath);
            std::filesystem::copy(std::filesystem::path(data.filePath) , std::filesystem::path(tempFilePath) , std::filesystem::copy_options::overwrite_existing);

            saveToDisk(data.filePath , diffStatus);

            return {};
        }

        VkDevice device = nullptr;

        std::string getCacheFileName(const std::string& info)
        {
            std::string fileName{};
            std::string filePath;
            parseFilePath(info , filePath , fileName);

            return "../ShaderModuleCache/" + fileName + ".bin";
        }

        [[nodiscard]] ErrorDataOutput<void> compileShaderObject(const CreateInfo &data , std::vector<VkDescriptorSetLayout> layouts) noexcept {
                std::string fileName{} , filePath{};
                parseFilePath(data.filePath , filePath , fileName);
                {                
                    const auto result = compileShader(filePath , fileName);
                    if(!result.has_value()) return result;
                }
                auto buffer = getSPIR_V_code(getCompiledShaderPath(fileName , filePath));
                if(buffer.empty()){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the cache buffer is nulltr"});
                }

                {
                    // only to check if its works
                    const ErrorDataType message{MOX_ERROR_TYPE_NONE , std::format("file || {} || needs to be recompiled , proceed" , data.filePath)};
                    engineLogger(message);
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
            return {};
        }

        std::string getTempFileName(const std::string& info)
        {
            std::string fileName{};
            std::string filePath;
            parseFilePath(info , filePath , fileName);

            return "../ShaderModuleCache/" + fileName + "Temp.hlsl";
        }

        std::string getCompiledShaderPath(const std::string fileName , const std::string filePath) noexcept
        {
            const std::string path = filePath + '/' + fileName + ".spv";
            return path;
        }


        // returns false if there are changes
        bool checkValidTemp(const std::string file) noexcept{
            auto path = getTempFileName(file);

            if(!std::filesystem::exists(path)) return false;
            if (std::filesystem::file_size(file) != std::filesystem::file_size(path))
                return false;   

            std::ifstream a(file , std::ios::binary);   // std::ios::binary - ТА ЖЕ причина, что для SPIR-V/pipeline cache!
            std::ifstream b(path , std::ios::binary);

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

        // checkValidShaderStructure and commitShaderStructureStamp are in RTCompileObjects ( the same for ShaderModule2 )

        [[nodiscard]] ErrorDataOutput<void> compileShader(const std::string path , const std::string fileName) noexcept {
            const std::string toCompile = path + '/' + fileName + ".hlsl";
            const std::string Fo =  path + '/' + fileName + ".spv";
            if(!std::filesystem::exists(toCompile)) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to compile the shader , since the file name is not valid")});
            if(std::filesystem::exists(Fo)){
                std::filesystem::remove(Fo);
            }
            
            const std::string cmd = getCompileCommand(toCompile , Fo);
            FILE* pipe = popen(cmd.c_str() , "r");
            if(!pipe) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to open the pipe for shader file : " + path});
            std::array<char , 1024> buffer{};
            std::string errorData{};
            // fgets puts the line and a terminating zero : only the line is taken ( the whole buffer has the rest of the earlier lines and zeros )
            while(fgets(buffer.data() , buffer.size() , pipe)){
                errorData += buffer.data();
            }

            if(pclose(pipe) != 0){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to compile shader {} : Error Message : {}" , fileName , errorData)});
            }
            return {};
        }

        std::string getCompileCommand(const std::string input , const std::string output) noexcept{
            const std::string directory = " -I ../include/ ";
            const std::string compile = "dxc -spirv -T lib_6_7 -HV 202x -fspv-target-env=vulkan1.3 -fspv-extension=SPV_EXT_descriptor_indexing -fspv-extension=SPV_KHR_ray_tracing -fspv-extension=SPV_KHR_ray_query ";
            const std::string cmd = compile  + directory + input + " -Fo " + output + " 2>&1";
            return cmd;
        }

        void parseFilePath(std::string_view fileFullPath, std::string& path, std::string& fileName) {
            std::filesystem::path p(fileFullPath);
            path = p.parent_path().string();
            fileName = p.stem().string();
        }

        void saveToDisk(const std::string &path , const bool diffStatus){
            if(!shader) return;
            if(diffStatus) return;
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

            // the copy of shaderStructure.h for the next run ( only when it changed in this one )
            commitShaderStructureStamp();
        }

    };

    using pShaderObject2 = std::shared_ptr<ShaderObject2>;
}


