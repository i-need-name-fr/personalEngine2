#pragma once

#include "Textures2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"

namespace mox{

    // one is buffer
    // two is sampler
    // three are textures

    class TextureManager2 final{
    private:
        const uint32_t MaxTextureCount = 9999u;

        std::vector<pTexture2> textures{};
        std::vector<int32_t> virtualToPhysical{};
        std::vector<int32_t> physicalToVirtual{};

        std::unordered_map<std::string , int32_t> pathToVirtual{};

        VkDevice device = nullptr;
        VulkanContext* context = nullptr;

        std::vector<DescriptorData2> getDescriptorData() const noexcept{

            std::vector<DescriptorData2> data;
            {
                const auto temp = DescriptorData2::createData(virtualBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0);
                data.push_back(temp);
            }

            {
                const auto temp = DescriptorData2::createData(typicalSampler.get() , 1 , 0);
                data.push_back(temp);
            }

            uint32_t i = 0 ;
            for(auto& texture : textures){
                const auto temp = DescriptorData2::createData(texture.get() , 2 , i);
                data.push_back(temp);
                i++;
            }
            
            return data;
        }

        ErrorDataOutput<void> updateData() noexcept {
            if(!virtualBuffer) return {};
            virtualBuffer->recordData(translateToSpan(virtualToPhysical));

            auto datas = getDescriptorData();

            return dbuffer->updateDescriptorSet(datas);
        }

        void performCopy(TextureManager2&& input) noexcept{
            textures = std::move(input.textures);
            virtualToPhysical = std::move(input.virtualToPhysical);
            physicalToVirtual = std::move(input.physicalToVirtual);

            layout = std::move(input.layout);
            dbuffer = std::move(input.dbuffer);
            virtualBuffer = std::move(input.virtualBuffer);
            typicalSampler = std::move(input.typicalSampler);

            device = std::exchange(input.device , nullptr);
            context = std::exchange(input.context , nullptr);
            pathToVirtual = std::move(input.pathToVirtual);
        }

        void performDelete() noexcept{
            layout = nullptr;
            virtualBuffer = nullptr;
            dbuffer = nullptr;
            typicalSampler = nullptr;
            layout = nullptr;
            device = nullptr;
            context = nullptr;
        }

        [[nodiscard]] [[gnu::cold]] ErrorDataOutput<void> init() noexcept{

            {
                std::vector<DescriptorLayoutData> data(3);
                data[0] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                data[1] = {.type = VK_DESCRIPTOR_TYPE_SAMPLER , .count = 1};
                data[2] = {.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , .count = MaxTextureCount};

                DescriptorLayout::CreateInfo info{};
                info.context = context;
                info.data = data;
                info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;
                info.stage = VK_SHADER_STAGE_ALL;

                layout = std::make_shared<DescriptorLayout>(info);
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
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(uint32_t) * Texture2::MaxTextureCount;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    virtualBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }
                catch(std::exception& e){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION,"failed to create host buffer for texture manager"});
                }

                try{
                    auto datas = getDescriptorData();
                    DescriptorBuffer2::CreateInfo info{};
                    info.data = datas;
                    info.context = context;
                    info.layout = layout.get();

                    dbuffer = std::make_shared<DescriptorBuffer2>(info);
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

        TextureManager2() = delete;
        TextureManager2& operator=(TextureManager2& input) = delete;
        TextureManager2(TextureManager2& input) = delete;
        TextureManager2& operator=(TextureManager2&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }

        TextureManager2(TextureManager2&& input) noexcept{
            performCopy(std::move(input));
        }

        ~TextureManager2(){
            performDelete();
        }

        pDescriptorBuffer2 dbuffer = nullptr;
        pGeneralBuffer2<BufferType::Device , true> virtualBuffer = nullptr;
        pSampler typicalSampler = nullptr;

        pDescriptorLayout layout = nullptr;

        explicit TextureManager2(const CreateInfo &info){
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

        [[nodiscard]] ErrorDataOutput<int32_t> addTexture(Texture2::CreateInfo info) noexcept{

            // the same file can be several textures only when a CHANNEL of it is a texture of its own ( R32_SFLOAT with a channel ) : then the key has the channel in it .
            // the RGBA8 maps keep all the channels , the same file is one texture whatever the channel in the info is ( the roughness and the metallic of an ARM map )
            const bool channelIsUsed = (info.format == VK_FORMAT_R32_SFLOAT && info.channel >= 0);
            const std::string key = channelIsUsed ? info.path + "#" + std::to_string(info.channel) : info.path;

            auto it = pathToVirtual.find(key);
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

                textures.push_back(std::move(std::make_shared<Texture2>(info)));
                pathToVirtual.insert(std::pair<std::string , uint32_t>(key , virtualIndex));

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

    using pTextureManager2 = std::shared_ptr<TextureManager2>;
}