#pragma once

#include "Textures.hpp"
#include "../DescriptorBuffer/Descriptor_buffer.hpp"

namespace mox{

    // one is buffer
    // two is sampler
    // three are textures

    class TextureManager final{
    private:

        std::vector<pTexture> textures{};
        std::vector<int32_t> virtualToPhysical{};
        std::vector<int32_t> physicalToVirtual{};

        std::unordered_map<std::string , int32_t> pathToVirtual{};

        VkDevice device = nullptr;
        VulkanContext* context = nullptr;

        std::vector<DescriptorData> getDescriptorData() const noexcept{
            std::vector<DescriptorData> datas{};
            // buffer
            {   
                DescriptorData info{};
                info.ASinfo = {};
                info.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                info.imageInfo = {};
                info.bindingIndex = 0;
                info.indexInArray = 0;
                info.bufferInfo= {virtualBuffer->pBuffer , virtualBuffer->getBufferSize()};
                info.samplerInfo = {};

                datas.push_back(info);
            }

            // sampler
            {
                DescriptorData info{};
                info.ASinfo = {};
                info.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                info.imageInfo = {};
                info.bindingIndex = 1;
                info.indexInArray = 0;
                info.bufferInfo= {};
                info.samplerInfo = {typicalSampler->sampler};

                datas.push_back(info);
            }
            uint32_t i = 0;
            for(auto& texture : textures){

                if(!texture) continue;

                DescriptorTypeImage image{};
                image.view = texture->view;

                DescriptorData info{};
                info.ASinfo = {};
                info.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                info.imageInfo = {image};
                info.bindingIndex = 2;
                info.indexInArray = i;
                info.bufferInfo= {};
                info.samplerInfo = {};
                i++;
                datas.push_back(info);
            }
            return datas;
        }

        ErrorDataOutput<void> updateData() noexcept {
            if(!virtualBuffer) return {};
            virtualBuffer->recordData(virtualToPhysical);

            auto datas = getDescriptorData();

            auto result = dbuffer->updateDescriptorBuffer(layout , datas);
            return result;
        }

        void performCopy(TextureManager&& input) noexcept{
            textures = std::move(input.textures);
            virtualToPhysical = std::move(input.virtualToPhysical);
            physicalToVirtual = std::move(input.physicalToVirtual);

            layout = std::exchange(input.layout, nullptr);
            dbuffer = std::move(input.dbuffer);
            virtualBuffer = std::move(input.virtualBuffer);
            typicalSampler = std::move(input.typicalSampler);

            device = std::exchange(input.device , nullptr);
            context = std::exchange(input.context , nullptr);
            pathToVirtual = std::move(input.pathToVirtual);
        }

        void performDelete() noexcept{
            if(device && layout) vkDestroyDescriptorSetLayout(device , layout , nullptr);
            virtualBuffer = nullptr;
            dbuffer = nullptr;
            typicalSampler = nullptr;
            layout = nullptr;
            device = nullptr;
            context = nullptr;
        }

        [[nodiscard]] [[gnu::cold]] ErrorDataOutput<void> init() noexcept{

            {
                std::vector<VkDescriptorSetLayoutBinding> bind(3);

                bind[0].binding = 0;
                bind[0].descriptorCount = 1;
                bind[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bind[0].pImmutableSamplers = nullptr;
                bind[0].stageFlags = VK_SHADER_STAGE_ALL;

                bind[1].binding = 1;
                bind[1].descriptorCount = 1;
                bind[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                bind[1].pImmutableSamplers = nullptr;
                bind[1].stageFlags = VK_SHADER_STAGE_ALL;

                bind[2].binding = 2;
                bind[2].descriptorCount = MaxTextureCount;
                bind[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                bind[2].pImmutableSamplers = nullptr;
                bind[2].stageFlags = VK_SHADER_STAGE_ALL;

                // descriptor buffers have no pool and no allocation of a set : UPDATE_AFTER_BIND needs the
                // UPDATE_AFTER_BIND_POOL layout flag , which a DESCRIPTOR_BUFFER layout must not have ( 08002 ) ,
                // and a variable count has nowhere to be given. only PARTIALLY_BOUND is needed : the slots of
                // textures that were not added yet are never written and never read.
                std::vector<VkDescriptorBindingFlags> flags = {0 , 0 , VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};
                
                VkDescriptorSetLayoutBindingFlagsCreateInfo ext{};
                ext.pBindingFlags = flags.data();
                ext.bindingCount = 3;
                ext.pNext = nullptr;
                ext.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;

                VkDescriptorSetLayoutCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                info.pNext = &ext;
                info.bindingCount = 3;
                info.pBindings = bind.data();
                // NOT a push descriptor layout : a push layout has an opaque size ( vkGetDescriptorSetLayoutSizeEXT
                // is invalid for it and returned 0 ) and is capped at maxPushDescriptors ( 32 ) descriptors
                info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;

                const auto result = vkCreateDescriptorSetLayout(device , &info , nullptr , &layout);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create descriptor set layout for texture manager , error ID -> {}" , static_cast<int32_t>(result))});
                }
            }

            {
                try{
                    Sampler::CreateInfo info{};
                    info.requiredComparingSampler = false;
                    info.requiredTypicalSampler = false;
                    info.textureSampler = true;
                    info.borderColorIsWhite = true;
                    info.context = context;
                    typicalSampler = std::make_shared<Sampler>(info);
                }
                catch(std::exception& e){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION,"failed to create typical sampler for texture manager"});
                }

                try{
                    HostBuffer::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(uint32_t) * MaxTextureCount;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    virtualBuffer = std::make_shared<HostBuffer>(info);
                }
                catch(std::exception& e){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION,"failed to create host buffer for texture manager"});
                }

                try{
                    auto datas = getDescriptorData();

                    DescriptorBuffer::CreateInfo info{};
                    info.layout = layout;
                    info.context = context;
                    info.descriptorInfo = datas;

                    dbuffer = std::make_shared<DescriptorBuffer>(info);
                }
                catch(std::exception& e){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION,"failed to create descriptor buffer for texture manager"});
                }
            }

            return {};
        }


        int32_t findVirtualIndex() noexcept {
            for(uint32_t i = 0 ; i < virtualToPhysical.size() ; i++){
                if(virtualToPhysical[i] == NON_VALID_SLOT) return i;
            }
            return NON_VALID_SLOT;
        }

    public:

        struct CreateInfo{
            VulkanContext* context;
        };

        TextureManager() = delete;
        TextureManager& operator=(TextureManager& input) = delete;
        TextureManager(TextureManager& input) = delete;
        TextureManager& operator=(TextureManager&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }

        TextureManager(TextureManager&& input) noexcept{
            performCopy(std::move(input));
        }

        ~TextureManager(){
            performDelete();
        }

        pDescriptorBuffer dbuffer = nullptr;
        pHostBuffer virtualBuffer = nullptr;
        pSampler typicalSampler = nullptr;

        VkDescriptorSetLayout layout = nullptr;

        explicit TextureManager(const CreateInfo &info){
            if(!info.context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create texture manager , since the context is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }
            if(!info.context->logicalDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create texture manager , since the logical device is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            this->context = info.context;
            this->device = info.context->logicalDevice->device;

            const auto result = init();
            if(!result.has_value()){
                engineLogger(&result.error());
                performDelete();
                THROW_MESSAGE;
            }
        }

        [[nodiscard]] ErrorDataOutput<int32_t> addTexture(Texture::CreateInfo info) noexcept{

            auto it = pathToVirtual.find(info.path);
            if(it != pathToVirtual.end()){
                return it->second;
            }

            try{
                int32_t virtualIndex = NON_VALID_SLOT;
                int32_t physicalIndex = textures.size();

                auto temp = findVirtualIndex();
                if(temp == NON_VALID_SLOT){

                    temp = virtualToPhysical.size();
                    virtualToPhysical.push_back(physicalIndex);
                    virtualIndex = temp;
                }else{
                    virtualIndex = temp;
                }

                physicalToVirtual.push_back(virtualIndex);
                virtualToPhysical[virtualIndex] = physicalIndex;

                textures.push_back(std::move(std::make_shared<Texture>(info)));
                pathToVirtual.insert(std::pair<std::string , uint32_t>(info.path , virtualIndex));

                auto result = updateData();
                if(!result.has_value()) return std::unexpected(result.error());
                return virtualIndex;
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create texture for texture manager : Texture path ->" , info.path)});
            }
        }

        ErrorDataOutput<void> deleteTexture(const std:: string path) noexcept{
            if(textures.empty()) return {};
            if(path.empty()) return{};
            auto it = pathToVirtual.find(path);
            if(it == pathToVirtual.end()) return{};

            const uint32_t virtualIndex = it->second;
            const uint32_t physicalIndex = virtualToPhysical[virtualIndex];

            if(physicalIndex == NON_VALID_SLOT) return {};

            for(uint32_t i = physicalIndex + 1 ; i < textures.size() ; i++){
                const uint32_t cVirtualIndex = physicalToVirtual[i];
                virtualToPhysical[cVirtualIndex]--;
            }

            virtualToPhysical[virtualIndex] = NON_VALID_SLOT;
            if(device) vkDeviceWaitIdle(device);

            textures.erase(textures.begin() + physicalIndex);
            physicalToVirtual.erase(physicalToVirtual.begin() + physicalIndex);

            auto result = updateData();
            if(!result.has_value()) return std::unexpected(result.error());

            pathToVirtual.erase(path);

            return {};
        }
    };

    using pTextureManager = std::shared_ptr<TextureManager>;
}