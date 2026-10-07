#pragma once

#include "../vulkanContext/vulkanBase.hpp"
#include <fstream>
#include <filesystem>
#include "../DescriptorBuffer/Descriptor_buffer.hpp"

namespace mox{

    class Pipeline;
    using pPipeline = std::shared_ptr<Pipeline>;

    class Pipeline{
    public:
        Pipeline() = delete;
        Pipeline(Pipeline& input) = delete;
        Pipeline(Pipeline&& input) noexcept{
            performCopy(std::move(input));
        }
        Pipeline& operator=(Pipeline& input) = delete;
        Pipeline& operator=(Pipeline&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context;
            std::string cacheFile;
        };

        VkPipeline pipeline = nullptr;
        VkRenderPass renderPass = nullptr;
        VkPipelineLayout layout = nullptr;

        std::vector<VkDescriptorSetLayout> d_layouts{};
        std::vector<pDescriptorBuffer> buffers{};


        Pipeline(const CreateInfo &info){
            if(!info.context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "Failed to create pipeline , since the context is nullptr"};
                THROW_MESSAGE;
            }

            this->device = info.context->logicalDevice->device;
            this->cacheSaveFile = info.cacheFile;
            cache = nullptr;
        }

        VkPipelineCache loadPipelineCache() noexcept{
            if(device){
                if(cache) return cache;
                if(cacheSaveFile.empty()) return nullptr;
                auto buffer = readFile(cacheSaveFile);
                const auto result = initCache(buffer);

                if(result.has_value()){
                    return cache;
                }else{
                    engineLogger(&result.error());
                    return nullptr;
                }
            }else return nullptr;
        }

        ~Pipeline() noexcept {
            if(device){
                if(pipeline) vkDestroyPipeline(device , pipeline , nullptr);
                if(layout ) vkDestroyPipelineLayout(device , layout , nullptr);
                if(renderPass) vkDestroyRenderPass(device , renderPass , nullptr);
                for(auto& d : d_layouts){
                    if(d) vkDestroyDescriptorSetLayout(device , d , nullptr);
                }
                if(cache){
                    saveTheCache();
                    vkDestroyPipelineCache(device , cache , nullptr);
                }
            }
            buffers.clear();
            d_layouts.clear();
            cache = nullptr;
            layout = nullptr;
            pipeline = nullptr;
            device = nullptr;
            renderPass = nullptr;
        }

    protected:

        void performCopy(Pipeline&& input) noexcept{
            pipeline = std::exchange(input.pipeline , nullptr);
            renderPass = std::exchange(input.renderPass , nullptr);
            layout = std::exchange(input.layout, nullptr);

            d_layouts = std::move(input.d_layouts);
            cache = std::exchange(input.cache , nullptr);
            cacheSaveFile = input.cacheSaveFile;
            device = std::exchange(input.device , nullptr);
            buffers = std::move(input.buffers);
        }

        VkDevice device = nullptr;

    private:

        VkPipelineCache cache = nullptr;
        std::string cacheSaveFile{};

        
        [[nodiscard]] std::vector<int8_t> readFile(const std::string path) noexcept{
            if(path.empty()){
                return {};
            }

            if(!std::filesystem::exists(path)) return {};
            
            std::ifstream file(path , std::ios::binary | std::ios::ate);
            if(!file.is_open()){
                return {};
            }

            std::size_t size = file.tellg();
            file.seekg(0);
            std::vector<int8_t> buffer(size);
            file.read(std::bit_cast<char*>(buffer.data()) , size);
            file.close();
            return buffer;
        }

        [[nodiscard]] ErrorDataOutput<void> initCache(std::vector<int8_t> &buffer) noexcept{
            VkPipelineCacheCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
            info.flags = 0; 
            if(buffer.empty()){
                info.initialDataSize = 0;
                info.pInitialData = nullptr;
            }else{
                info.initialDataSize = static_cast<uint32_t>(buffer.size());
                info.pInitialData = buffer.data();
            }
            info.pNext = nullptr;

            const auto result = vkCreatePipelineCache(device , &info , nullptr , &cache);
            if(result != VK_SUCCESS){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create pipeline cache \nfile path ->{} \nError ID ->{} ", cacheSaveFile , static_cast<int32_t>(result))});
            }

            return {};
        }

        void saveTheCache() noexcept {
            if(!cache) return;
            std::ofstream file(cacheSaveFile , std::ios::binary | std::ios::out);
            if(file.is_open()){
                std::size_t cacheSize;
                vkGetPipelineCacheData(device , cache , &cacheSize , nullptr);
                std::vector<uint8_t> bufferData(cacheSize);
                vkGetPipelineCacheData(device , cache , &cacheSize , bufferData.data());
                file.write(std::bit_cast<const char*>(bufferData.data()) , cacheSize);
                file.close();
            }
        }
    };  

    using pPipeline = std::shared_ptr<Pipeline>;
}