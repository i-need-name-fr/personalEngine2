#pragma once

#include "../Buffers/buffer.hpp"
#include "../Image2/RenderImage2.hpp"
#include "../Image2/Skybox2.hpp"
#include "../Image2/Textures2.hpp"
#include "../Sampler/Sampler.hpp"
#include <variant>

namespace mox{

    class DescriptorBuffer;
    using pDescriptorBuffer = std::shared_ptr<DescriptorBuffer>;

    struct DescriptorTypeImage{

        VkImageView view= nullptr;
        
    };

    struct DescriptorTypeSampler{
        VkSampler sampler;
    };  

    struct DescriptorTypeAS{
        VkDeviceAddress pAS;
    };

    struct DescriptorTypeBuffer{
        VkDeviceAddress pBuffer;
        // VK_WHOLE_SIZE is not allowed next to a device address , the real size of the buffer is required
        VkDeviceSize range = 0;
    };

    struct DescriptorData{
        VkDescriptorType descriptorType;

        uint32_t bindingIndex;
        uint32_t indexInArray = 0;

        DescriptorTypeImage imageInfo{};
        DescriptorTypeAS ASinfo{};
        DescriptorTypeSampler samplerInfo{};
        DescriptorTypeBuffer bufferInfo{};

    };

    inline DescriptorData createDescriptorData(VkImageView view, VkDescriptorType descriptorType, const uint32_t index , const uint32_t arID) noexcept{
        
        DescriptorTypeImage image{};
        image.view = view;

        DescriptorData data{};
        data.descriptorType = descriptorType;
        data.ASinfo = {};
        data.samplerInfo = {};
        data.bufferInfo = {};
        data.imageInfo = image;
        data.bindingIndex = index;
        data.indexInArray = arID;
        return data;
    }   

    inline DescriptorData createDescriptorData(BufferBase* buffer , VkDescriptorType descriptorType, const uint32_t index , const uint32_t arID) noexcept{
        if(!buffer) return {};

        DescriptorData data{};

        data.ASinfo = {};
        data.descriptorType= descriptorType;
        data.bufferInfo = {buffer->pBuffer , buffer->getBufferSize()};
        data.imageInfo ={};
        data.samplerInfo = {};
        data.bindingIndex = index;
        data.indexInArray = arID;
        return data;
    }

    inline DescriptorData createDescriptorData(Sampler* sampler, const uint32_t index , const uint32_t arID) noexcept{

        if(!sampler) return {};

        DescriptorData data{};

        data.ASinfo = {};
        data.descriptorType= VK_DESCRIPTOR_TYPE_SAMPLER;
        data.bufferInfo = {};
        data.imageInfo ={};
        data.samplerInfo = {sampler->sampler};
        data.bindingIndex = index;
        data.indexInArray = arID;
        return data;
    }

    inline DescriptorData createDescriptorData(VkDeviceAddress pAS, const uint32_t index , const uint32_t arID){

        DescriptorData data{};

        data.ASinfo = {pAS};
        data.descriptorType= VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        data.bufferInfo = {};
        data.imageInfo ={};
        data.samplerInfo = {};
        data.bindingIndex = index;
        data.indexInArray = arID;
        return data;
    }

    inline DescriptorData createDescriptorData(const std::variant<Sampler* , VkDeviceAddress , Skybox2* , Texture2* , RenderImage2* , BufferBase*> input , VkDescriptorType type , const uint32_t index , const uint32_t arID){
        DescriptorData data{};

        if(std::holds_alternative<Sampler*>(input)){

            const auto sampler = std::get<Sampler*>(input);
            if(!sampler){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the sampler is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            } 
            data = createDescriptorData(sampler , index , arID);

        }else if(std::holds_alternative<Skybox2*>(input)){

            const auto image = std::get<Skybox2*>(input);
            if(!image){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the skybox2 is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            if(!image->view){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the skybox2 view is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            data = createDescriptorData(image->view , type , index , arID);

        }else if(std::holds_alternative<RenderImage2*>(input)){

            const auto image = std::get<RenderImage2*>(input);
            if(!image){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the render image2 is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            if(!image->view){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the render image2 view is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            data = createDescriptorData(image->view , type , index , arID);

        }else if(std::holds_alternative<Texture2*>(input)){

            const auto image = std::get<Texture2*>(input);
            if(!image){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the texture2 is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            if(!image->view){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the texture2 view is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            data = createDescriptorData(image->view , type , index , arID);

        }else if(std::holds_alternative<BufferBase*>(input)){

            const auto buffer = std::get<BufferBase*>(input);
            if(!buffer){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the buffer is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            if(!buffer->buffer){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the buffers buffer is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            data = createDescriptorData(buffer , type , index , arID);

        }else if(std::holds_alternative<VkDeviceAddress>(input)){

            const auto AS = std::get<VkDeviceAddress>(input);
            if(!AS){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor data , since the AS address is nullptr , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
                return {};
            }
            data = createDescriptorData(AS , index , AS);

        }else{
            return {};
        }

        return data;
    }

    template<typename T>
    concept DescriptorDataType = std::is_same_v<T , DescriptorTypeImage> ||std::is_same_v<T , DescriptorTypeBuffer>  || std::is_same_v<T , DescriptorTypeAS> || std::is_same_v<T , DescriptorTypeSampler> ;

    // descriptor buffer will be one per set, so the input data should be in the same order , as its in the shader
    class DescriptorBuffer final : public BufferBase{
    public:

        DescriptorBuffer() = delete;
        DescriptorBuffer(DescriptorBuffer& input) = delete;
        DescriptorBuffer(DescriptorBuffer&& input) noexcept : BufferBase(std::move(input)){}
        DescriptorBuffer& operator=(DescriptorBuffer& input) = delete;
        DescriptorBuffer& operator=(DescriptorBuffer&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{

            VulkanContext* context;
            VkDescriptorSetLayout layout;
            std::vector<DescriptorData> descriptorInfo;

        };

        DescriptorBuffer(const CreateInfo &info) : BufferBase(info.context){

            if(info.descriptorInfo.empty()){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , " failed to create descriptor set , since the sets are null"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            VkDeviceSize size;
            getDescriptorSetLayoutSizeEXT(device , info.layout , &size);

            {
                BufferBase::BufferBaseCreateInfo data{};
                data.context = context;
                data.memoryProperty = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                data.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT ;
                std::cout << std::format("size is - > {}" , size);
                data.size = size;

                const auto result = createBuffer(data);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }

                {
                    const auto result = vkMapMemory(device , memory , 0 , size ,0 , &this->data);
                    if(result != VK_SUCCESS){
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create descriptor buffer , map memory error , Error ID -> {}", static_cast<int32_t>(result))};
                        engineLogger(&error);
                        THROW_MESSAGE;
                    }
                }
            }

            for(uint32_t i = 0 ; i < info.descriptorInfo.size() ;i++){
                const auto result = processBinding(info.layout , info.descriptorInfo[i]);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }

        }

        [[nodiscard]] ErrorDataOutput<void> updateDescriptorBuffer(const VkDescriptorSetLayout layout , const std::vector<DescriptorData> &data) noexcept{
            if(!layout) return {};
            if(data.empty()) return {};
            for(auto& input : data){
                const auto result = processBinding(layout , input);
                if(!result.has_value()) return std::unexpected(result.error());
            }
            
            return {};
        }



        ~DescriptorBuffer(){}

    private:

        inline static VkPhysicalDeviceDescriptorBufferPropertiesEXT properties{};
        inline static bool loadedProperties = false;

        [[nodiscard]] ErrorDataOutput<void> processBinding(const VkDescriptorSetLayout layout ,const DescriptorData &data){

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
            getDescriptorSetLayoutBindingOffsetEXT(device , layout , data.bindingIndex , &offset);

            uint8_t* ptr = std::bit_cast<uint8_t*>(this->data);
            if(!ptr){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "couldnt process binding , since the buffer hasnt been created"});
            }
            ptr += offset;

            if(data.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || data.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER){

                VkDescriptorAddressInfoEXT address{};
                address.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT;
                address.format = VK_FORMAT_UNDEFINED;
                if(!data.bufferInfo.pBuffer){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since buffer address is not valid"});
                }
                if(data.bufferInfo.range == 0){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since the buffer range is 0 ( VK_WHOLE_SIZE is not allowed with an address )"});
                }
                address.address = data.bufferInfo.pBuffer;
                address.range = data.bufferInfo.range;
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
                if(!data.samplerInfo.sampler){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since the sampler is null"});
                }
                info.data.pSampler = &data.samplerInfo.sampler;
                ptr += data.indexInArray * properties.samplerDescriptorSize;
                getDescriptorEXT(device , &info , properties.samplerDescriptorSize , ptr);

            }else if(data.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR){

                VkDescriptorGetInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
                info.type = data.descriptorType;
                info.pNext = nullptr;
                if(!data.ASinfo.pAS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to write to descriptor buffer , since the acceleration structure is null"});
                }
                info.data.accelerationStructure = data.ASinfo.pAS;
                ptr += data.indexInArray * properties.accelerationStructureDescriptorSize;

                getDescriptorEXT(device , &info , properties.accelerationStructureDescriptorSize , ptr);

            }else if(data.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || data.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE){

                VkDescriptorImageInfo image{};
                image.sampler = nullptr;
                image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                image.imageView = data.imageInfo.view;

                VkDescriptorGetInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
                info.type = data.descriptorType;
                info.pNext = nullptr;
                if(!data.imageInfo.view){
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

    };
}