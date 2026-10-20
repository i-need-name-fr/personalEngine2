#pragma once

#include "../vulkanContext/vulkanBase.hpp"
#include <filesystem>
#include "../RTCompileObjects/RTCompileObjects.hpp"

namespace mox{

    class ShaderModule2 final : public RTCompileObjects{
    private:

        std::string getTempFileName(const std::string& info)
        {
            std::string fileName{};
            std::string filePath;
            parseFilePath(info , filePath , fileName);

            return "../PipelineCache/" + fileName + "Temp.hlsl";
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

            std::ifstream a(file , std::ios::binary);   
            std::ifstream b(path , std::ios::binary);

            return std::equal(
                std::istreambuf_iterator<char>(a),
                std::istreambuf_iterator<char>(),
                std::istreambuf_iterator<char>(b)
            );
        }

        // checkValidShaderStructure and commitShaderStructureStamp are in RTCompileObjects ( the same for ShaderObject2 )

        void parseFilePath(std::string_view fileFullPath, std::string& path, std::string& fileName) {
            std::filesystem::path p(fileFullPath);
            path = p.parent_path().string();
            fileName = p.stem().string();
        }

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

        void saveToDisk(const std::string path){
            const std::string temp = getTempFileName(path);
            try{
                std::filesystem::copy_file(path ,temp , std::filesystem::copy_options::overwrite_existing);
            }
            catch(std::filesystem::filesystem_error& e){
                std::cerr << "failed to reupdate the HLSL file of this shader moduleb : " << e.what() << '\n';
            }

            // the copy of shaderStructure.h for the next run ( only when it changed in this one )
            commitShaderStructureStamp();
        }

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

        [[nodiscard]] ErrorDataOutput<void> compileAndCreateModule(const std::string path) noexcept {

            if(!std::filesystem::exists(path)) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader module , since the path is not valid : file path {}" , path)});

            // compiling the shader
            std::string fileName{} , filePath;
            parseFilePath(path , filePath , fileName);
            const std::string outputFile = filePath + '/' + fileName + ".spv";
            {
                const auto result = compileShader(filePath , fileName);
                if(!result.has_value()) return result;
            }


            // after the compilation , we need to create the shader module itself

            auto buffer = readFile(outputFile);
            if(buffer.empty()){
                ErrorDataType const error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader module of file - >  {}" , path)};
                return std::unexpected(error);
            }

            {
                // only to check if its works
                const ErrorDataType message{MOX_ERROR_TYPE_NONE , std::format("file || {} || needs to be recompiled , proceed" , outputFile)};
                engineLogger(message);
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

        ErrorDataOutput<void> createShaderModule(const std::string path) noexcept {
            if(!std::filesystem::exists(path)) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to compile the shader , since the file name is not valid -> Path: {}", path)});
            const bool diffStatus = checkValidTemp(path) && checkValidShaderStructure();
            if(diffStatus){
                std::string fileName{} , filePath;
                parseFilePath(path , filePath , fileName);
                const std::string outputFile = filePath + '/' + fileName + ".spv";
                auto buffer = readFile(outputFile);
                if(buffer.empty()){
                    ErrorDataType const error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader module of file , however will try to recompile it- >  {}" , path)};
                    engineLogger(error);
                    const auto result = compileAndCreateModule(path);
                    if(!result.has_value()) return result;
                    saveToDisk(path);
                    return {};
                }

                VkShaderModuleCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                info.codeSize = static_cast<uint32_t>(buffer.size());
                info.flags = 0;
                info.pNext = nullptr;
                info.pCode = std::bit_cast<const uint32_t*>(buffer.data()); 

                auto result = vkCreateShaderModule(device , &info , nullptr , &module);
                if(result != VK_SUCCESS){
                    ErrorDataType const error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader module PATH -> {}, however will try to recompile it : Error ID -> {}" , path , static_cast<int32_t>(result))};
                    engineLogger(error);
                    const auto result = compileAndCreateModule(path);
                    if(!result.has_value()) return result;
                    saveToDisk(path);
                    return {};
                }

            }else{

                const auto result = compileAndCreateModule(path);
                if(!result.has_value()) return result;

            }

            saveToDisk(path);
            return {};
        }

        VkDevice device = nullptr;

        void clean() noexcept {
            if(device){
                if(module) vkDestroyShaderModule(device , module , nullptr);
            }

            module = nullptr;
            device = nullptr;
        }

        void processCopy(ShaderModule2&& input) noexcept{
            module = std::exchange(input.module,  nullptr);
            device = std::exchange(input.device , nullptr);
        }


    public:

        VkShaderModule module = nullptr;

        SetTypicalOperators(ShaderModule2)
        SetTypicalTransferOperators(ShaderModule2)

        ShaderModule2(VulkanContext* context , const std::string path){
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

        ~ShaderModule2(){clean();}
    };

    using pShaderModule2 = std::shared_ptr<ShaderModule2>;

}