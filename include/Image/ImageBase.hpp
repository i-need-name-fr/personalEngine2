#pragma once
#include "../vulkanContext/vulkanBase.hpp"
#include <stb/stb_image.h>

namespace mox{

    class ImageBase;
    using pImageBase = std::shared_ptr<ImageBase>;

    // base will only contain the image , view port has to be defined in other classes
    // image base is the power source of render image and Texture
    // it doesnt have the image view, since it doesnt need it

    // 
    class ImageBase{
    public:

        ImageBase() = delete;
        ImageBase(ImageBase& input ) = delete;

        ImageBase(VulkanContext* context){
            [[unlikely]] if(!context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create image base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            [[unlikely]] if(!context->queueManager){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create image base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            [[unlikely]] if(!context->logicalDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create image base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            this->context = context;
            this->device = context->logicalDevice->device;
        }
        ImageBase(ImageBase&& input) noexcept{
            performCopy(std::move(input));
        }

        ImageBase& operator=(ImageBase& input) = delete;
        ImageBase& operator=(ImageBase&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }

        virtual ~ImageBase(){
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

        VkExtent2D extent;
        VkFormat format;

        VkImageMemoryBarrier2 translateImage(VkImageLayout src , VkImageLayout dst ,const uint32_t layerCount , const uint32_t layerIndex , const uint32_t mipCount , const uint32_t mipIndex ) noexcept{
            VkImageMemoryBarrier2 barrier{};

            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.pNext = nullptr;
            barrier.image =this->image;
            barrier.subresourceRange.aspectMask = generalImageData.aspect;
            barrier.subresourceRange.levelCount = mipCount;
            barrier.subresourceRange.layerCount = layerCount;
            barrier.subresourceRange.baseArrayLayer = layerIndex;
            barrier.subresourceRange.baseMipLevel = mipIndex;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            switch(src){
                case VK_IMAGE_LAYOUT_UNDEFINED :{
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = 0;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL :{
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : {
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : {
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = 0;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_GENERAL : {
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : {
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : {
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : {
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                    break;
                }
                default : {
                    barrier.oldLayout = src;
                    barrier.srcAccessMask = 0;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    break;
                }
            }

            // every branch here used to assign `src` to newLayout, which made every
            // barrier in the whole renderer a no-op: oldLayout == newLayout.
            switch(dst){
                case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL :{
                    barrier.newLayout = dst;
                    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : {
                    barrier.newLayout = dst;
                    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : {
                    barrier.newLayout = dst;
                    barrier.dstAccessMask = 0;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_GENERAL : {
                    barrier.newLayout = dst;
                    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : {
                    barrier.newLayout = dst;
                    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : {
                    barrier.newLayout = dst;
                    barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                    break;
                }
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : {
                    barrier.newLayout = dst;
                    barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                    break;
                }
                default : {
                    barrier.newLayout = dst;
                    barrier.dstAccessMask = 0;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    break;
                }
            }
            return barrier;
        }

        // NOTE on allLayouts vs virtualLayouts :
        // - allLayouts     tracks the REAL VkImageLayout the image is in on the GPU. under the unified
        //                   layout scheme every barrier below ends with the image physically in GENERAL,
        //                   so this always converges to GENERAL after the first call.
        // - virtualLayouts tracks the LOGICAL role the image was last used for (e.g. COLOR_ATTACHMENT_OPTIMAL,
        //                   SHADER_READ_ONLY_OPTIMAL). translateImage() only uses this to pick the right
        //                   access/stage masks for the *previous* usage - it never reaches the driver.
        VkImageMemoryBarrier2 translateUnified(VkImageLayout src , VkImageLayout dst ,const uint32_t layerCount , const uint32_t layerIndex , const uint32_t mipCount , const uint32_t mipIndex ) noexcept{
            const uint64_t index = layerIndex * generalImageData.mipMapCount + mipIndex;
            auto barrier =translateImage( src, dst , layerCount , layerIndex , mipCount ,mipIndex);
            const auto currentLayout = allLayouts[index];
            barrier.oldLayout = currentLayout;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

            // used to set allLayouts -> GENERAL and then immediately overwrite the same indices with `dst`,
            // so the tracked "real" layout silently became `dst` even though the GPU image was actually put
            // into GENERAL. the next call then read `dst` back as oldLayout, which no longer matched what
            // the image was really in - a real VUID mismatch, not just a missed optimization.
            for (uint32_t l = layerIndex; l < layerIndex + layerCount; l++) {
                for (uint32_t m = mipIndex; m < mipIndex + mipCount; m++) {
                    const uint64_t idx = l * generalImageData.mipMapCount + m;
                    allLayouts[idx] = VK_IMAGE_LAYOUT_GENERAL;
                    virtualLayouts[idx] = dst;
                }
            }
            return barrier;
        }

        VkImageMemoryBarrier2 translateUnifiedAuto(VkImageLayout dst ,const uint32_t layerCount , const uint32_t layerIndex , const uint32_t mipCount , const uint32_t mipIndex ) noexcept{
            // getImageLayout() reads virtualLayouts, which used to never get written anywhere - every
            // auto call picked up VK_IMAGE_LAYOUT_UNDEFINED as `src`, so translateImage() always produced
            // srcAccessMask = 0 / srcStageMask = TOP_OF_PIPE, i.e. no real wait on the previous usage.
            const auto src = getImageLayout(layerCount , layerIndex , mipCount , mipIndex);
            const uint64_t index = layerIndex * generalImageData.mipMapCount + mipIndex;
            auto barrier =translateImage( src, dst , layerCount , layerIndex , mipCount ,mipIndex);
            const auto currentLayout = allLayouts[index];
            barrier.oldLayout = currentLayout;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

            for (uint32_t l = layerIndex; l < layerIndex + layerCount; l++) {
                for (uint32_t m = mipIndex; m < mipIndex + mipCount; m++) {
                    const uint64_t idx = l * generalImageData.mipMapCount + m;
                    allLayouts[idx] = VK_IMAGE_LAYOUT_GENERAL;
                    virtualLayouts[idx] = dst;
                }
            }
            return barrier;
        }

        VkImageLayout getImageLayout(const uint32_t layerCount , const uint32_t layerIndex , const uint32_t mipCount , const uint32_t mipIndex ) const noexcept{
            return virtualLayouts[layerIndex * generalImageData.mipMapCount + mipIndex];
        }

        uint32_t getImageMipCount() const noexcept{
            return generalImageData.mipMapCount;
        }

        uint32_t getImageLayerCount() const noexcept{
            return generalImageData.layerCount;
        }

    protected:

        VulkanContext* context;
        VkDevice device = nullptr;

        ImageInfo generalImageData{};

        std::vector<VkImageLayout> allLayouts{};
        std::vector<VkImageLayout> virtualLayouts{};

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

            {
                const uint64_t layoutSize = generalImageData.mipMapCount * generalImageData.layerCount;
                allLayouts = std::vector<VkImageLayout>(layoutSize , VK_IMAGE_LAYOUT_UNDEFINED);
                virtualLayouts = std::vector<VkImageLayout>(layoutSize , VK_IMAGE_LAYOUT_UNDEFINED);
            }

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
        
        void performCopy(ImageBase&& input) noexcept{
            device = std::exchange(input.device , nullptr);
            context = std::exchange(input.context , nullptr);
            image = std::exchange(input.image , nullptr);
            memory = std::exchange(input.memory , nullptr);
            extent = input.extent;
            format = input.format;
            generalImageData = input.generalImageData;
            allLayouts = std::move(input.allLayouts);
            virtualLayouts = std::move(input.virtualLayouts);
        }



    };

}