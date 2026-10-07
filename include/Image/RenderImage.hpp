#pragma once

#include "ImageBase.hpp"

namespace mox{

    class RenderImage;
    using pRenderImage = std::shared_ptr<RenderImage>;

    // render image is designed for storage images
    // that are used in shaders or pipelines
    // instead of textures : this rely on extent of swapchain

    class RenderImage final : public ImageBase{
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

            bool viewPerLayer = false; // for CSM maybe idk
            bool viewPerMip = false; // for frustum 

            VkClearColorValue clear{};
        };

        RenderImage() = delete;
        RenderImage(RenderImage& input) = delete;
        RenderImage(RenderImage&& input) noexcept : ImageBase(std::move(input)){
            view = std::exchange(input.view , nullptr);
            viewPerMip = std::move(input.viewPerMip);
            viewPerLayer = std::move(input.viewPerLayer);
        }


        RenderImage& operator=(RenderImage& input) = delete;
        RenderImage& operator=(RenderImage&& input) noexcept{
            performCopy(std::move(input));
            view = std::exchange(input.view , nullptr);
            viewPerMip = std::move(input.viewPerMip);
            viewPerLayer = std::move(input.viewPerLayer);
            return *this;
        }

        ~RenderImage() override{
            cleanRenderImage();
        }

        explicit RenderImage(const CreateInfo &data) : ImageBase(data.context){
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
                const auto result = translateToFinalLayout(data.finalLayout , data.clear);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    cleanRenderImage();
                    THROW_MESSAGE;
                }
            }
        }

        VkImageView view = nullptr;
        std::vector<VkImageView> viewPerMip{};
        std::vector<VkImageView> viewPerLayer{};

    private:

        void cleanRenderImage() noexcept{
            if(device){
                if(view) vkDestroyImageView(device , view , nullptr);
                for(auto& v : viewPerMip) if(v) vkDestroyImageView(device , v , nullptr);
                for(auto& v : viewPerLayer) if(v) vkDestroyImageView(device , v , nullptr);
            }
            view = nullptr;
            viewPerMip.clear();
            viewPerLayer.clear();
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

            if(data.viewPerMip){
                viewPerMip.resize(data.mipMapCount);
                for(uint32_t i = 0 ; i < data.mipMapCount ; i++){
                    const auto result = createImageView(viewPerMip[i] , data.layerCount , 0 , 1 , i);
                    if(!result.has_value()){
                        return std::unexpected(result.error());
                    }
                }
            }

            if(data.viewPerLayer){
                viewPerLayer.resize(data.layerCount);
                for(uint32_t i = 0 ; i < data.layerCount ; i++){
                    const auto result = createImageView(viewPerLayer[i] , 1 , i , data.mipMapCount , 0);
                    if(!result.has_value()){
                        return std::unexpected(result.error());
                    }
                }
            }

            return{};
        }

        [[nodiscard]] ErrorDataOutput<void> translateToFinalLayout(VkImageLayout finalLayout , const VkClearColorValue value ){

            cmdFunctionType recordFunction = [&] (VkCommandBuffer &cmd)  ->void{
                // one barrier covers every mip and every layer , the per mip / per layer views only look at
                // subresources of this image , so they are in the right layout as soon as the whole image is
                auto barrier = translateUnified(VK_IMAGE_LAYOUT_UNDEFINED , VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL , generalImageData.layerCount , 0 , generalImageData.mipMapCount , 0);

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

                if(format != VK_FORMAT_D32_SFLOAT){
                    vkCmdClearColorImage(cmd , image , VK_IMAGE_LAYOUT_GENERAL , &value , 1 , &range);
                }else{
                    VkClearDepthStencilValue depthValue{};
                    depthValue.depth = 1.0f;
                    depthValue.stencil = 0;

                    vkCmdClearDepthStencilImage(cmd , image , VK_IMAGE_LAYOUT_GENERAL , &depthValue , 1 , &range);
                }

                barrier =  translateUnified(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL , finalLayout , generalImageData.layerCount , 0 , generalImageData.mipMapCount , 0);

                info.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd , &info);
            };

            // the lambda captures finalLayout and value by reference , so it has to run before this function returns
            context->queueManager->Begin();

            context->queueManager->recordCmd(recordFunction , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

            const auto result = context->queueManager->End();
            if(!result.has_value()){
                return std::unexpected(result.error());
            }

            return {};
        }

    };

}