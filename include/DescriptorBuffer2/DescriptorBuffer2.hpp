#pragma once

#include "../Buffer2/BufferBase2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../Image2/RenderImage2.hpp"
#include "../Image2/Skybox2.hpp"
#include "../Image2/Textures2.hpp"
#include <variant>
#include "../AS2/AccelerationStructureBase.hpp"
#include "../Sampler/Sampler.hpp"

namespace mox{


    struct DescriptorData2{
        VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_MAX_ENUM;

        uint32_t bindingIndex =0;
        uint32_t indexInArray = 0;

        struct {
            BufferBase2* buffer{nullptr};
            VkImageView imageView{nullptr};
            AccelerationStructure* tlas{nullptr};
            Sampler* sampler{nullptr};
        } data;

        static DescriptorData2 createData(BufferBase2* buffer , VkDescriptorType type , const uint32_t binding , const uint32_t ID) noexcept{
            if(!buffer) return {};
            if(!buffer->buffer) return {};
            DescriptorData2 data{};
            data.bindingIndex = binding;
            data.descriptorType = type;
            data.indexInArray = ID;
            data.data.buffer = buffer;
            return data;
        }

        static DescriptorData2 createData(AccelerationStructure* tlas, const uint32_t binding , const uint32_t ID) noexcept{
            if(!tlas) return {};
            if(!tlas->pAS) return {};
            DescriptorData2 data{};
            data.bindingIndex = binding;
            data.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            data.indexInArray = ID;
            data.data.tlas = tlas;
            return data;
        }

        static DescriptorData2 createData(Sampler* sampler , const uint32_t binding , const uint32_t ID) noexcept{
            if(!sampler) return {};
            if(!sampler->sampler) return {};
            DescriptorData2 data{};
            data.bindingIndex = binding;
            data.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            data.indexInArray = ID;
            data.data.sampler = sampler;
            return data;
        }

        static DescriptorData2 createData(std::variant<RenderImage2* , Skybox2*> image , VkDescriptorType type , const uint32_t binding , const uint32_t ID) noexcept{
            DescriptorData2 data{};
            data.bindingIndex = binding;
            data.descriptorType = type;
            data.indexInArray = ID;
            if(std::holds_alternative<RenderImage2*>(image)){

                RenderImage2* input = std::get<RenderImage2*>(image);
                if(!input) return {};
                if(!input->view) return {};
                data.data.imageView = input->view;

            }else{

                Skybox2* input = std::get<Skybox2*>(image);
                if(!input) return {};
                if(!input->view) return {};
                data.data.imageView = input->view;

            }
            return data;
        }

        static DescriptorData2 createData(Texture2* image , const uint32_t binding , const uint32_t ID) noexcept{
            if(!image) return {};
            if(!image->view) return {};
            DescriptorData2 data{};
            data.bindingIndex = binding;
            data.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            data.indexInArray = ID;
            data.data.imageView = image->view;
            return data;
        }

        static DescriptorData2 createData(VkImageView image , VkDescriptorType type, const uint32_t binding , const uint32_t ID) noexcept{
            if(!image) return {};
            DescriptorData2 data{};
            data.bindingIndex = binding;
            data.descriptorType = type;
            data.indexInArray = ID;
            data.data.imageView = image;
            return data;
        }
    };


    class DescriptorBuffer2 final : public BufferBase2{
    public:

        DescriptorBuffer2() = delete;
        DescriptorBuffer2(const DescriptorBuffer2& input) = delete;
        DescriptorBuffer2(DescriptorBuffer2& input) = delete;
        DescriptorBuffer2& operator=(DescriptorBuffer2& input) = delete;
        DescriptorBuffer2& operator=(const DescriptorBuffer2& input) = delete;

        DescriptorBuffer2(DescriptorBuffer2&& input) noexcept : BufferBase2(std::move(input)){
            layout = std::exchange(input.layout , nullptr);
            inputData = std::move(input.inputData);
            input.inputData.clear();
        }
        DescriptorBuffer2& operator=(DescriptorBuffer2&& input) noexcept{
            if(this == &input) return *this;
            // performCopy only takes the handles of the other buffer , the buffer and the memory of this one would stay alive for ever
            cleanBuffer();
            performCopy(std::move(input));
            layout = std::exchange(input.layout , nullptr);
            inputData = std::move(input.inputData);
            input.inputData.clear();
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context{nullptr};
            DescriptorLayout* layout{nullptr};
            std::vector<DescriptorData2> data{};
        };

        explicit DescriptorBuffer2(const CreateInfo& info) : BufferBase2(info.context){

            if(!info.layout){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create DesciptorBuffer2 , since the descriptor layout is nullptr"});
                THROW_MESSAGE;
            }

            if(!info.layout->layout){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create DesciptorBuffer2 , since the descriptor layout is nullptr"});
                THROW_MESSAGE;
            }

            layout = info.layout;

            {
                VkDeviceSize size = 0;
                getDescriptorSetLayoutSizeEXT(info.context->logicalDevice->device , info.layout->layout , &size);
                BufferBaseCreateInfo data{};
                data.context = info.context;
                data.memoryProperty = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                data.size = size;
                data.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                const auto result = createBuffer(data);
                if(!result.has_value()){
                    engineLogger(result.error());
                    cleanBuffer();
                    layout = nullptr;
                    THROW_MESSAGE;
                }

                const auto mapped = vkMapMemory(info.context->logicalDevice->device , memory , 0 , size , 0 , &this->data);
                if(mapped != VK_SUCCESS){
                    engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create DescriptorBuffer2 , map memory error , Error ID -> {}" , static_cast<int32_t>(mapped))});
                    cleanBuffer();
                    layout = nullptr;
                    THROW_MESSAGE;
                }
            }

            for(auto& data : info.data){
                const auto result = processBinding(info.layout , data);
                if(!result.has_value()){
                    engineLogger(result.error());
                    cleanBuffer();
                    layout = nullptr;
                    THROW_MESSAGE;
                }
            }
            inputData = info.data;
        }

        [[nodiscard]] ErrorDataOutput<void> updateDescriptorSet(const DescriptorData2 &data){

            if(!layout) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_WARNING , "couldnt update the descriptor buffer2 , since the layout is empty"});

            if(!checkSimilarity(data)) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "couldnt update the descriptor buffer2 , since the input data type is different from the prev one"});

            return processBinding(layout , data);
        }

        ~DescriptorBuffer2()noexcept{
            inputData.clear();
            layout = nullptr;
        }

    private:

        inline static VkPhysicalDeviceDescriptorBufferPropertiesEXT properties{};
        inline static bool loadedProperties = false;

        std::vector<DescriptorData2> inputData{};

        bool checkSimilarity(const DescriptorData2 data) const noexcept{
            for(auto& d : inputData){

                if(data.bindingIndex == d.bindingIndex && data.indexInArray == d.indexInArray){
                    if(d.descriptorType != data.descriptorType) return false;
                }
            }

            return true;
        }

        [[nodiscard]] ErrorDataOutput<void> processBinding(const DescriptorLayout* layout ,const DescriptorData2 &data){

            if(!loadedProperties){
                properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;

                static VkPhysicalDeviceProperties2 deviceProperties{};
                deviceProperties.pNext = &properties;
                deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                deviceProperties.properties = {};

                vkGetPhysicalDeviceProperties2(context->physicalDevice->device , &deviceProperties);
                loadedProperties = true;
            }

            VkDeviceSize offset;
            getDescriptorSetLayoutBindingOffsetEXT(device , layout->layout , data.bindingIndex , &offset);

            uint8_t* ptr = std::bit_cast<uint8_t*>(this->data);
            if(!ptr){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "couldnt process binding , since the buffer hasnt been created"});
            }
            ptr += offset;

            if(data.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || data.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER){

                VkDescriptorAddressInfoEXT address{};
                address.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT;
                address.format = VK_FORMAT_UNDEFINED;
                if(!data.data.buffer->pBuffer){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since buffer address is not valid"});
                }
                if(data.data.buffer->getBufferSize() == 0){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since the buffer range is 0 ( VK_WHOLE_SIZE is not allowed with an address )"});
                }
                address.address = data.data.buffer->pBuffer;
                address.range = data.data.buffer->getBufferSize();
                address.pNext = nullptr;

                VkDescriptorGetInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
                info.type = data.descriptorType;
                info.pNext = nullptr;

                if(data.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER){
                    info.data.pUniformBuffer = &address;
                    ptr += data.indexInArray * properties.uniformBufferDescriptorSize;
                    getDescriptorEXT(device , &info , properties.uniformBufferDescriptorSize , ptr);
                }else{
                    info.data.pStorageBuffer = &address;
                    ptr += data.indexInArray * properties.storageBufferDescriptorSize;
                    getDescriptorEXT(device , &info , properties.storageBufferDescriptorSize , ptr);
                }

            }else if(data.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER){

                VkDescriptorGetInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
                info.type = data.descriptorType;
                info.pNext = nullptr;
                if(!data.data.sampler){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since the sampler is null"});
                }
                info.data.pSampler = &data.data.sampler->sampler;
                ptr += data.indexInArray * properties.samplerDescriptorSize;
                getDescriptorEXT(device , &info , properties.samplerDescriptorSize , ptr);

            }else if(data.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR){

                VkDescriptorGetInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
                info.type = data.descriptorType;
                info.pNext = nullptr;
                if(!data.data.tlas->pAS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since the acceleration structure is null"});
                }
                info.data.accelerationStructure = data.data.tlas->pAS;
                ptr += data.indexInArray * properties.accelerationStructureDescriptorSize;

                getDescriptorEXT(device , &info , properties.accelerationStructureDescriptorSize , ptr);

            }else if(data.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || data.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE){

                VkDescriptorImageInfo image{};
                image.sampler = nullptr;
                image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                image.imageView = data.data.imageView;

                VkDescriptorGetInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
                info.type = data.descriptorType;
                info.pNext = nullptr;
                if(!data.data.imageView){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since the image view  is null"});
                }

                if(data.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE){
                    info.data.pSampledImage = &image;
                    ptr += data.indexInArray * properties.sampledImageDescriptorSize;

                    getDescriptorEXT(device , &info , properties.sampledImageDescriptorSize , ptr);
                }else{
                    info.data.pStorageImage = &image;
                    ptr += data.indexInArray * properties.storageImageDescriptorSize;
                    getDescriptorEXT(device , &info , properties.storageImageDescriptorSize , ptr);
                }

            }else{
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write in descriptor buffer , since the type of DescriptorType is undefined"});
            }
            
            return {};
        }

        DescriptorLayout* layout{nullptr};
    };

    using pDescriptorBuffer2 = std::shared_ptr<DescriptorBuffer2>;
}


