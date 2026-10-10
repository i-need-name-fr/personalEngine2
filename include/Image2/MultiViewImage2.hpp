#pragma once

#include "ImageBase2.hpp"

namespace mox{

    enum class ViewPerType{
        Layer,
        Mip
    };

    // =============================== INFO ===============================
    // MultiViewImage2 : ONE image with `count` slices ( layers or mips ) , every slice has its own view and its own state ( access and stage ) ,
    // so a pass can write one slice while another pass reads a different one , and the barriers know what happened to every slice
    //
    // views[i]  : the view of the slice i ( only that layer or only that mip )
    // fullView  : the view of the WHOLE image : all the layers ( 2D array ) , or all the mips ( the same view type as the slices ) ,
    //             for reading it as an array / a mip chain after the slices were written
    //
    // translateImage( access , stage , count , index ) : the barrier for the slices [ index , index + count ) , the state of every one
    //                                                    of them is updated , the source masks are the union of their old states
    // translateWholeImage( access , stage )            : the same for all the slices ( for fullView )
    // =============================== END ===============================

    template<uint32_t count , ViewPerType type>
    requires(count >= 2)
    class MultiViewImage2 final : public ImageBase2{
    public:

        MultiViewImage2() = delete;
        MultiViewImage2(MultiViewImage2& input) = delete;
        MultiViewImage2(const MultiViewImage2& input) = delete;
        MultiViewImage2& operator=(MultiViewImage2& input) = delete;
        MultiViewImage2& operator=(const MultiViewImage2& input) = delete;

        MultiViewImage2(MultiViewImage2&& input) noexcept : ImageBase2(std::move(input)){
            takeViews(std::move(input));
        }

        MultiViewImage2& operator=(MultiViewImage2&& input) noexcept{
            if(this == &input) return *this;
            // the views and the image that this object held would leak
            destroyViews();
            cleanImage();
            performCopy(std::move(input));
            takeViews(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context;

            VkFormat imageFormat;
            VkImageUsageFlags imageUsage;
            VkExtent2D extent;
            VkImageAspectFlags aspect;
            VkImageViewType viewType;
            VkImageType imageType;

            VkClearColorValue clear{};
        };

        explicit MultiViewImage2(const CreateInfo &info) : ImageBase2(info.context) {
            {
                const auto result = createImageViews(info);
                if(!result.has_value()){
                    engineLogger(result.error());
                    // the destructor does not run when the constructor throws
                    destroyViews();
                    cleanImage();
                    THROW_MESSAGE;
                }
            }
        }

        std::array<VkImageView , count> views{};
        VkImageView fullView{nullptr};

        VkImageMemoryBarrier2 translateImage(const VkAccessFlags2 access ,const VkPipelineStageFlags2 stage , const uint32_t layerCount , const uint32_t layerIndex) noexcept requires(type == ViewPerType::Layer){
            uint32_t first = 0 , amount = 0;
            clampSlices(layerIndex , layerCount , first , amount , "layer");

            VkImageMemoryBarrier2 barrier = makeBarrier(access , stage , first , amount);
            barrier.subresourceRange.baseArrayLayer = first;
            barrier.subresourceRange.layerCount = amount;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            return barrier;
        }

        VkImageMemoryBarrier2 translateImage(const VkAccessFlags2 access ,const VkPipelineStageFlags2 stage , const uint32_t mipCount , const uint32_t mipIndex) noexcept requires(type == ViewPerType::Mip){
            uint32_t first = 0 , amount = 0;
            clampSlices(mipIndex , mipCount , first , amount , "mip");

            VkImageMemoryBarrier2 barrier = makeBarrier(access , stage , first , amount);
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.baseMipLevel = first;
            barrier.subresourceRange.levelCount = amount;
            return barrier;
        }

        VkImageMemoryBarrier2 translateWholeImage(const VkAccessFlags2 access ,const VkPipelineStageFlags2 stage) noexcept{
            return translateImage(access , stage , count , 0);
        }

        VkAccessFlags2 acquireSliceAccess(const uint32_t index) const noexcept{
            return index < count ? accesses[index] : VK_ACCESS_2_NONE;
        }

        VkPipelineStageFlags2 acquireSliceStage(const uint32_t index) const noexcept{
            return index < count ? stages[index] : VK_PIPELINE_STAGE_2_NONE;
        }

        ~MultiViewImage2(){
            destroyViews();
        }

    private:

        std::array<VkAccessFlags2 , count> accesses{};
        std::array<VkPipelineStageFlags2 , count> stages{};

        void destroyViews() noexcept{
            if(device){
                for(auto& view : views){
                    if(view) vkDestroyImageView(device , view , nullptr);
                }
                if(fullView) vkDestroyImageView(device , fullView , nullptr);
            }
            views.fill(nullptr);
            fullView = nullptr;
        }

        void takeViews(MultiViewImage2&& input) noexcept{
            views = input.views;
            input.views.fill(nullptr);
            fullView = std::exchange(input.fullView , nullptr);
            accesses = input.accesses;
            stages = input.stages;
        }

        void clampSlices(const uint32_t index , const uint32_t amount , uint32_t &first , uint32_t &clampedAmount , const char* what) const noexcept{
            first = std::min(index , count - 1);
            clampedAmount = std::min(std::max(amount , 1u) , count - first);

            [[unlikely]] if(first != index || clampedAmount != amount){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("MultiViewImage2 translateImage : the {} range [{} , +{}) is out of the {} slices of the image , it is cut to [{} , +{}) , message Count {} " , what , index , amount , count , first , clampedAmount , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }
        }

        VkImageMemoryBarrier2 makeBarrier(const VkAccessFlags2 access , const VkPipelineStageFlags2 stage , const uint32_t first , const uint32_t amount) noexcept{
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.pNext = nullptr;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange.aspectMask = generalImageData.aspect;
            barrier.image = image;
            barrier.dstAccessMask = access;
            barrier.dstStageMask = stage;

            for(uint32_t i = first ; i < first + amount ; i++){
                barrier.srcAccessMask |= accesses[i];
                barrier.srcStageMask |= stages[i];

                accesses[i] = access;
                stages[i] = stage;
            }
            return barrier;
        }

        [[nodiscard]] ErrorDataOutput<void> createView(const CreateInfo &input , VkImageView &view , const VkImageViewType viewType , const uint32_t baseLayer , const uint32_t layers , const uint32_t baseMip , const uint32_t mips) noexcept{
            VkImageViewCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            info.pNext = nullptr;
            info.flags = 0;
            info.viewType = viewType;
            info.format = input.imageFormat;
            info.image = image;
            info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            info.subresourceRange.aspectMask = input.aspect;
            info.subresourceRange.baseArrayLayer = baseLayer;
            info.subresourceRange.layerCount = layers;
            info.subresourceRange.baseMipLevel = baseMip;
            info.subresourceRange.levelCount = mips;

            const auto result = vkCreateImageView(context->logicalDevice->device , &info , nullptr , &view);
            if(result != VK_SUCCESS){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create an image view for MultiViewImage2 , layer {} + {} , mip {} + {} , error ID -> {}" , baseLayer , layers , baseMip , mips , static_cast<int32_t>(result))});
            }
            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> createImageViews(const CreateInfo &input) noexcept{

            ImageInfo info{};
            info.imageFlags = 0;
            info.imageFormat = input.imageFormat;
            info.imageType = input.imageType;
            info.viewType = input.viewType;
            info.imageUsage = input.imageUsage | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.aspect = input.aspect;
            info.context = input.context;
            info.extent = input.extent;

            if constexpr (type == ViewPerType::Layer){
                info.mipMapCount = 1;
                info.layerCount = count;
            }else{
                info.layerCount = 1;
                const uint32_t maxMip = static_cast<uint32_t>(std::floor(std::log2(std::max(input.extent.width , input.extent.height)))) + 1;
                if(count > maxMip){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create MultiViewImage2 , since {} mips were asked , but an image {} x {} has only {}" , count , input.extent.width , input.extent.height , maxMip)});
                }
                info.mipMapCount = count;
            }

            {
                const auto result = createImage(info);
                if(!result.has_value()) return result;
            }

            for(uint32_t i = 0 ; i < count ; i++){
                ErrorDataOutput<void> result{};
                if constexpr (type == ViewPerType::Layer){
                    result = createView(input , views[i] , input.viewType , i , 1 , 0 , 1);
                }else{
                    result = createView(input , views[i] , input.viewType , 0 , 1 , i , 1);
                }
                if(!result.has_value()) return result;
            }

            {
                ErrorDataOutput<void> result{};
                if constexpr (type == ViewPerType::Layer){
                    result = createView(input , fullView , VK_IMAGE_VIEW_TYPE_2D_ARRAY , 0 , count , 0 , 1);
                }else{
                    result = createView(input , fullView , input.viewType , 0 , 1 , 0 , count);
                }
                if(!result.has_value()) return result;
            }

            cmdFunctionType function = [&](VkCommandBuffer &cmd) -> void{

                auto barrier = translateWholeImage(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                VkDependencyInfo dependency{};
                dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dependency.bufferMemoryBarrierCount = 0;
                dependency.imageMemoryBarrierCount = 1;
                dependency.pMemoryBarriers = nullptr;
                dependency.pBufferMemoryBarriers = nullptr;
                dependency.pImageMemoryBarriers = &barrier;
                dependency.memoryBarrierCount = 0;
                dependency.pNext = nullptr;
                dependency.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &dependency);

                // and one range clears all of it
                VkImageSubresourceRange range{};
                range.aspectMask = generalImageData.aspect;
                range.baseArrayLayer = 0;
                range.baseMipLevel = 0;

                if constexpr (type == ViewPerType::Layer){
                    range.layerCount = count;
                    range.levelCount = 1;
                }else{
                    range.levelCount = count;
                    range.layerCount = 1;
                }

                // i will add later other formats
                if(format != VK_FORMAT_D32_SFLOAT){

                    vkCmdClearColorImage(cmd , image , VK_IMAGE_LAYOUT_GENERAL , &input.clear , 1 , &range);
                }else{
                    VkClearDepthStencilValue depthValue{};
                    depthValue.depth = 1.0f;
                    depthValue.stencil = 0;

                    vkCmdClearDepthStencilImage(cmd , image , VK_IMAGE_LAYOUT_GENERAL , &depthValue , 1 , &range);
                }

                // the same state as RenderImage2 has after its creation : what was cleared is visible to every other use
                barrier = translateWholeImage(VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                dependency.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd , &dependency);
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

    template<uint32_t count , ViewPerType type>
    requires(count >= 2)
    using pMultiViewImage2 = std::shared_ptr<MultiViewImage2<count , type>>;
}
