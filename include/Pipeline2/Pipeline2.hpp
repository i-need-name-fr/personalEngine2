#pragma once

#include "../vulkanContext/vulkanBase.hpp"
#include <fstream>
#include <filesystem>
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../PipelineLayout/PipelineLayout.hpp"
#include "../ShaderModule2/ShaderModule2.hpp"

namespace mox{

    class Pipeline2;
    using pPipeline2 = std::shared_ptr<Pipeline2>;

    class Pipeline2{
    public:

        SetTypicalOperators(Pipeline2)
        SetTypicalTransferOperators(Pipeline2)

        struct CreateInfo{
            VulkanContext* context;
            std::string cacheFile;
        };

        VkPipeline pipeline = nullptr;
        VkRenderPass renderPass = nullptr;
        pPipelineLayout layout = nullptr;

        std::vector<pDescriptorLayout> descriptorLayouts{};
        std::vector<pDescriptorBuffer2> buffers{};


        Pipeline2(const CreateInfo &info){
            if(!info.context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "Failed to create pipeline , since the context is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            this->device = info.context->logicalDevice->device;
            this->cacheSaveFile = info.cacheFile;
            cache = nullptr;
            context = info.context;
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

        std::vector<VkDescriptorSetLayout> getDescriptorLayoutData() const noexcept{
            if(descriptorLayouts.empty()) return {};
            std::vector<VkDescriptorSetLayout> data{};
            for(auto& l : descriptorLayouts){
                data.push_back(l->layout);
            }
            return data;
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

        ~Pipeline2() noexcept {
            releaseResources();
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

    protected:

        // everything this object owns is destroyed ( the cache is saved first ) : the destructor and the move use it
        void releaseResources() noexcept {
            if(device){
                if(pipeline) vkDestroyPipeline(device , pipeline , nullptr);
                if(renderPass) vkDestroyRenderPass(device , renderPass , nullptr);
                if(cache){
                    saveTheCache();
                    vkDestroyPipelineCache(device , cache , nullptr);
                }
            }
            buffers.clear();
            descriptorLayouts.clear();
            cache = nullptr;
            layout = nullptr;
            pipeline = nullptr;
            device = nullptr;
            renderPass = nullptr;
            layout = nullptr;
            context = nullptr;
        }

        void processCopy(Pipeline2&& input) noexcept{
            // a move assignment used to take the handles of the other pipeline and leave the ones of this pipeline alive for ever ,
            // and a move of an object into itself emptied its descriptor layouts
            if(this == &input) return;
            releaseResources();
            pipeline = std::exchange(input.pipeline , nullptr);
            renderPass = std::exchange(input.renderPass , nullptr);
            layout =std::move(input.layout);

            descriptorLayouts = std::move(input.descriptorLayouts);
            cache = std::exchange(input.cache , nullptr);
            cacheSaveFile = input.cacheSaveFile;
            device = std::exchange(input.device , nullptr);
            buffers = std::move(input.buffers);
            context = std::exchange(input.context, nullptr);
        }

        VkDevice device = nullptr;
        VulkanContext* context;

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

}