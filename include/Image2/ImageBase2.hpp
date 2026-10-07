#pragma once
#include "../vulkanContext/vulkanBase.hpp"
#include <stb/stb_image.h>

namespace mox{

    class ImageBase2;
    using pImageBase2 = std::shared_ptr<ImageBase2>;

    // base will only contain the image , view port has to be defined in other classes
    // image base is the power source of render image and Texture
    // it doesnt have the image view, since it doesnt need it

    // 
    class ImageBase2{
    public:

        ImageBase2() = delete;
        ImageBase2(ImageBase2& input ) = delete;

        ImageBase2(VulkanContext* context){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager , context);

            if(!result.has_value()){
                engineLogger(result.error());
                THROW_MESSAGE;
            }

            this->context = context;
            this->device = context->logicalDevice->device;
        }
        ImageBase2(ImageBase2&& input) noexcept{
            performCopy(std::move(input));
        }

        ImageBase2& operator=(ImageBase2& input) = delete;
        ImageBase2& operator=(ImageBase2&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }

        virtual ~ImageBase2(){
            cleanImage();
        }

        struct ImageInfo{
            VulkanContext* context;
            
            VkImageViewType viewType;
            VkImageCreateFlags imageFlags;
            VkFormat imageFormat;
            VkImageUsageFlags imageUsage;
            VkImageType imageType;
            VkExtent2D extent;
            VkImageAspectFlags aspect;

            uint32_t layerCount;
            uint32_t mipMapCount;
            
        };

        VkImage image = nullptr;

        VkDeviceMemory memory = nullptr;

        VkExtent2D extent = {0 ,0};
        VkFormat format{VK_FORMAT_UNDEFINED};

    protected:

        VulkanContext* context = nullptr;
        VkDevice device = nullptr;

        ImageInfo generalImageData{};

        void cleanImage(){
            if(device){
                if(image) vkDestroyImage(device , image , nullptr);
                if(memory) vkFreeMemory(device , memory , nullptr);
            }

            image = nullptr;
            memory = nullptr;
            device = nullptr;
            context = nullptr;
        }

        [[nodiscard]] [[gnu::cold]] ErrorDataOutput<void> createImage(const ImageInfo& data){
            if(!context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the image , since the context has null parameters"};
                return std::unexpected(error);
            }

            if(!context->queueManager){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the image , since the context has null parameters"};
                return std::unexpected(error);
            }

            if(!device){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the image , since the context has null parameters"};
                return std::unexpected(error);
            }

            {
                VkImageCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                info.flags = data.imageFlags;
                info.pNext = nullptr;
                info.mipLevels = data.mipMapCount;
                info.arrayLayers = data.layerCount;
                info.extent = {data.extent.width , data.extent.height , 1};
                info.format = data.imageFormat;
                info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                info.imageType = data.imageType;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.sharingMode =VK_SHARING_MODE_EXCLUSIVE;
                info.usage=  data.imageUsage;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;

                const auto result = vkCreateImage(device , &info  ,nullptr , &image);
                if(result != VK_SUCCESS){
                    const ErrorDataType error{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create image , error ID -> {}" , static_cast<int32_t>(result))};
                    return std::unexpected(error);
                }

            }

            {
                VkMemoryRequirements req{};
                vkGetImageMemoryRequirements(device , image , &req);

                VkMemoryAllocateInfo info{};
                info.memoryTypeIndex = findMemoryIndex(context, req.memoryTypeBits , VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                info.pNext = nullptr;
                info.allocationSize = req.size;

                const auto result = vkAllocateMemory(device , &info , nullptr , &memory);
                if(result != VK_SUCCESS){
                    const ErrorDataType error{MOX_ERROR_TYPE_CRITICAL , std::format("failed to allocate the memory , error ID -> {}" , static_cast<int32_t>(result))};
                    return std::unexpected(error);
                }
            }

            const auto result = vkBindImageMemory(device , image , memory , 0);
            if(result != VK_SUCCESS){
                const ErrorDataType error{MOX_ERROR_TYPE_CRITICAL , std::format("failed to bind the image memory , error ID -> {}" , static_cast<int32_t>(result))};
                return std::unexpected(error);
            }

            extent = data.extent;
            format = data.imageFormat;
            generalImageData = data;

            return {};
        }

        [[gnu::cold]] [[nodiscard]] ErrorDataOutput<void> createImageView(VkImageView &view , const uint32_t layerCount , const uint32_t layerIndex , const uint32_t mipCount , const uint32_t mipIndex ) noexcept {
            VkImageViewCreateInfo info{};
            info.sType =VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            info.image = this->image;
            info.viewType = generalImageData.viewType;
            info.pNext = nullptr;
            info.format = this->format;
            info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            info.subresourceRange.aspectMask = generalImageData.aspect;
            info.subresourceRange.baseArrayLayer = layerIndex;
            info.subresourceRange.baseMipLevel = mipIndex;
            info.subresourceRange.levelCount = mipCount;
            info.subresourceRange.layerCount = layerCount;

            const auto result = vkCreateImageView(device , &info , nullptr , &view);
            if(result != VK_SUCCESS){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create a specific image view, layer -> {} , mip -> {} . ERROR ID -> {}" , layerIndex , mipIndex,  static_cast<int32_t>(result))};
                return std::unexpected(error);
            }
            return {};
        }
        
        void performCopy(ImageBase2&& input) noexcept{
            device = std::exchange(input.device , nullptr);
            context = std::exchange(input.context , nullptr);
            image = std::exchange(input.image , nullptr);
            memory = std::exchange(input.memory , nullptr);
            extent = input.extent;
            format = input.format;
            generalImageData = input.generalImageData;
        }



    };

}