#pragma once

#include "ImageBase2.hpp"

namespace mox{

    class RenderImage2;
    using pRenderImage2 = std::shared_ptr<RenderImage2>;

    // =============================== INFO ===============================
    // RenderImage2 
    // CHANGES:
    // new render image has only one view, since other will be created per mip or layer
    // now it has only 2 translates
    // first is translateImage , basically works same as old one
    // the only difference is the 2nd one
    // the second one translate image from undefined to general
    // the difference is that it can be used to reset image
    // since in the prev one we didnt care about it 
    // however basically nothing new tbh
    // =============================== END ===============================

    class RenderImage2 final : public ImageBase2{
    public:

        struct CreateInfo{
            VulkanContext* context;

            VkFormat imageFormat;
            VkImageUsageFlags imageUsage;
            VkExtent2D extent;
            VkImageAspectFlags aspect;
            VkImageViewType viewType;
            VkImageType imageType;

            VkImageLayout finalLayout;

            uint32_t layerCount = 1;
            uint32_t mipMapCount = 1;

            VkClearColorValue clear{};
        };

        RenderImage2() = delete;
        RenderImage2(RenderImage2& input) = delete;
        RenderImage2(RenderImage2&& input) noexcept : ImageBase2(std::move(input)){
            view = std::exchange(input.view , nullptr);
            currentAccess = input.currentAccess;
            currentStage = input.currentStage;
        }


        RenderImage2& operator=(RenderImage2& input) = delete;
        RenderImage2& operator=(RenderImage2&& input) noexcept{
            performCopy(std::move(input));
            view = std::exchange(input.view , nullptr);
            currentAccess = input.currentAccess;
            currentStage = input.currentStage;
            return *this;
        }

        ~RenderImage2() override{
            cleanRenderImage();
        }

        explicit RenderImage2(const CreateInfo &data) : ImageBase2(data.context){
            {
                const auto result = createTheRenderImage(data);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    // the destructor never runs when the constructor throws
                    cleanRenderImage();
                    THROW_MESSAGE;
                }
            }

            {
                const auto result = translateImageCreation(data);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    cleanRenderImage();
                    THROW_MESSAGE;
                }
            }
        }

        VkImageView view = nullptr;

        // translate image

        VkImageMemoryBarrier2 translateImage(const VkAccessFlags access ,const VkPipelineStageFlags stage) noexcept{
            VkImageMemoryBarrier2 barrier{};

            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.image = this->image;
            barrier.pNext = nullptr;
            barrier.srcAccessMask = currentAccess;
            barrier.dstAccessMask = access;
            barrier.srcStageMask = currentStage;
            barrier.dstStageMask = stage;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.subresourceRange.aspectMask = generalImageData.aspect;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.layerCount = generalImageData.layerCount;
            barrier.subresourceRange.levelCount = generalImageData.mipMapCount;

            currentAccess = access;
            currentStage = stage;

            return barrier;
        }

        // translate from undefined

        VkImageMemoryBarrier2 translateImageUndefined(const VkAccessFlags access ,const VkPipelineStageFlags stage) noexcept{
            VkImageMemoryBarrier2 barrier{};

            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.image = this->image;
            barrier.pNext = nullptr;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = access;
            barrier.srcStageMask = currentStage;
            barrier.dstStageMask = stage;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.subresourceRange.aspectMask = generalImageData.aspect;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.layerCount = generalImageData.layerCount;
            barrier.subresourceRange.levelCount = generalImageData.mipMapCount;

            currentAccess = access;
            currentStage = stage;

            return barrier;
        }


        const VkAccessFlags2 acquireImageAccess() const noexcept{
            return currentAccess;
        }

        const VkPipelineStageFlags2 acquireImageStage() const noexcept{
            return currentStage;
        }

    private:

        VkAccessFlags2 currentAccess = VK_ACCESS_2_NONE;
        VkPipelineStageFlags2 currentStage = VK_PIPELINE_STAGE_2_NONE;

        void cleanRenderImage() noexcept{
            if(device){
                if(view) vkDestroyImageView(device , view , nullptr);
            }
            view = nullptr;
        }

        [[nodiscard]] ErrorDataOutput<void> createTheRenderImage(const CreateInfo &data) noexcept{

            [[unlikely]] if(!context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create render image , since the context is null"};
                return std::unexpected(error);
            }

            [[unlikely]] if(!device){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create render image , since the device is null"};
                return std::unexpected(error);
            }

            ImageInfo info{};
            info.context = data.context;
            info.imageFlags = 0;
            info.imageFormat = data.imageFormat;
            info.imageType = data.imageType;
            info.imageUsage = data.imageUsage | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.viewType = data.viewType;
            info.aspect = data.aspect;
            info.extent = data.extent;
            info.layerCount = data.layerCount;
            info.mipMapCount = data.mipMapCount;

            {
                // creating the image first
                const auto result = createImage(info);
                if(!result.has_value()){
                    return std::unexpected(result.error());
                }
            }

            // now we create the image views
            {
                const auto result = createImageView(view , data.layerCount , 0 , data.mipMapCount , 0);
                if(!result.has_value()){
                    return std::unexpected(result.error());
                }
            }

            return{};
        }

        ErrorDataOutput<void> translateImageCreation(const CreateInfo &input) noexcept{

            currentAccess = 0;
            currentStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;


            cmdFunctionType function = [&](VkCommandBuffer &cmd) ->void {
                if(!cmd) return;

                auto barrier = translateImageUndefined(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 1;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = &barrier;
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);

                // and one range clears all of it
                VkImageSubresourceRange range{};
                range.aspectMask = generalImageData.aspect;
                range.baseArrayLayer = 0;
                range.baseMipLevel = 0;
                range.layerCount = generalImageData.layerCount;
                range.levelCount = generalImageData.mipMapCount;
                // i will add later other formats
                if(format != VK_FORMAT_D32_SFLOAT){

                    vkCmdClearColorImage(cmd , image , VK_IMAGE_LAYOUT_GENERAL , &input.clear , 1 , &range);
                }else{
                    VkClearDepthStencilValue depthValue{};
                    depthValue.depth = 1.0f;
                    depthValue.stencil = 0;

                    vkCmdClearDepthStencilImage(cmd , image , VK_IMAGE_LAYOUT_GENERAL , &depthValue , 1 , &range);
                }

                barrier = translateImage(VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                info.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd , &info);
            };

            context->queueManager->Begin();

            context->queueManager->recordCmd(function , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

            const auto result = context->queueManager->End();
            if(!result.has_value()){
                return std::unexpected(result.error());
            }

            return {};
        }
    };

    using pRenderImage2 = std::shared_ptr<RenderImage2>;
}