#pragma once

#include "ImageBase2.hpp"
#include "../Buffers/buffer.hpp"

namespace mox{

    class Texture2;
    using pTexture2 = std::shared_ptr<Texture2>;

    // simple texture , using stb
    class Texture2 final : public ImageBase2{
    public:

        Texture2() = delete;
        Texture2(Texture2&& input) noexcept: ImageBase2(std::move(input)){
            view = std::exchange(input.view , nullptr);
        }

        Texture2(Texture2& input) = delete;
        Texture2& operator=(Texture2& input) = delete;
        Texture2& operator=(Texture2&& input) noexcept{
            performCopy(std::move(input));
            view = std::exchange(input.view , nullptr);

            return *this;
        }
        
        struct CreateInfo{
            VulkanContext* context;

            VkFormat format;
            std::string path;
        };

        inline static constexpr uint32_t MaxTextureCount = 9999u;

        VkImageView view = nullptr;

        ~Texture2(){
            if(device && view) vkDestroyImageView(device , view , nullptr);
            view = nullptr;
        }
        
        Texture2(const CreateInfo &info) : ImageBase2(info.context){

            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager , info.context);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }

            this->device = context->logicalDevice->device;
            this->context = context;

            const auto result = processTexture(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                // the destructor of Texture never runs when the constructor throws
                if(view) vkDestroyImageView(device , view , nullptr);
                view = nullptr;
                THROW_MESSAGE;
            }
        }
    
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
            barrier.subresourceRange.layerCount = 1;
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
            barrier.subresourceRange.layerCount = 1;
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

        [[nodiscard]] ErrorDataOutput<void> processTexture(const CreateInfo &data) noexcept{
            if(!device){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create since info in context is nullptr , Texture Creation"};
                return std::unexpected(error);
            }

            if(!context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create since info in context is nullptr , Texture Creation"};
                return std::unexpected(error);
            }

            if(data.path.empty()){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the texture , since the path is empty");
                return std::unexpected(error);
            }

            int width = 0 , height = 0;
            void* readData = nullptr;
            VkDeviceSize size =0;
            if(data.format == VK_FORMAT_R32_SFLOAT){
                readData = stbi_loadf(data.path.c_str() , &width , &height , nullptr , 1);
                size = width * height * 4;
            }else if(data.format == VK_FORMAT_R8G8B8A8_UNORM){
                readData = stbi_load(data.path.c_str() , &width , &height , nullptr , 4);
                size = width * height * 4;
            }else if(data.format == VK_FORMAT_R8G8_UNORM){
                // stb : "2 channels" is grey + alpha , NOT red + green . a normal map came out as ( luminance , 1 ) : every normal was
                // tilted the same way and the light was wrong . load the four channels and keep red and green
                unsigned char* rgba = stbi_load(data.path.c_str() , &width , &height , nullptr , 4);
                if(rgba){
                    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
                    // malloc : readData is released with stbi_image_free ( free ) below
                    unsigned char* rg = static_cast<unsigned char*>(malloc(pixels * 2));
                    if(rg){
                        for(size_t i = 0 ; i < pixels ; i++){
                            rg[i * 2] = rgba[i * 4];
                            rg[i * 2 + 1] = rgba[i * 4 + 1];
                        }
                    }
                    stbi_image_free(rgba);
                    readData = rg;
                }
                size = static_cast<VkDeviceSize>(width) * height * 2;
            }
            // if we didnt find the texture
            if(!readData){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("couldnt create the texture -> |{}|,  since the path for image is nullptr OR undefined format" , data.path)};
                stbi_image_free(readData);
                return std::unexpected(error);
            }

            HostBuffer::CreateInfo bufferInfo{};
            bufferInfo.context = data.context;
            bufferInfo.size = size;
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

            HostBuffer buffer(bufferInfo, readData);

            stbi_image_free(readData);

            {
                ImageInfo info{};
                info.context = data.context;
                info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                info.imageType = VK_IMAGE_TYPE_2D;
                info.imageFlags = 0;
                info.imageFormat = data.format;
                info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                info.mipMapCount = (uint32_t)std::floor(std::log2(std::max(width, height))) + 1;
                info.extent = {static_cast<uint32_t>(width) ,static_cast<uint32_t>(height)};
                info.layerCount= 1;

                {
                    const auto result = createImage(info);
                    if(!result.has_value()){
                        return std::unexpected(result.error());
                    }
                }

                {
                    const auto result = createImageView(view , 1 , 0 , generalImageData.mipMapCount , 0);
                    if(!result.has_value()){
                        return std::unexpected(result.error());
                    }
                }
            }

            cmdFunctionType recordBuffer = [&](VkCommandBuffer &cmd) ->void{
                // translate first barrier
                {
                    auto barrier = translateImageUndefined(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
                    VkDependencyInfo dep{};
                    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.bufferMemoryBarrierCount = 0;
                    dep.imageMemoryBarrierCount = 1;
                    dep.pMemoryBarriers = nullptr;
                    dep.pBufferMemoryBarriers = nullptr;
                    dep.pImageMemoryBarriers = &barrier;
                    dep.memoryBarrierCount = 0;
                    dep.pNext = nullptr;
                    dep.dependencyFlags = 0;

                    vkCmdPipelineBarrier2(cmd , &dep);

                }

                // copy
                auto result = copyBufferToImage(cmd , buffer.buffer , image , size , extent);
                if(!result.has_value()) return;

                // generate mip maps
                int32_t width = extent.width;
                int32_t height = extent.height;

                VkImageMemoryBarrier2 barrier{};
                barrier.image = image;
                barrier.pNext = nullptr;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.baseArrayLayer = 0;
                barrier.subresourceRange.layerCount = 1;
                barrier.subresourceRange.levelCount = 1;

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.bufferMemoryBarrierCount = 0;
                dep.imageMemoryBarrierCount = 1;
                dep.pMemoryBarriers = nullptr;
                dep.pBufferMemoryBarriers = nullptr;
                dep.pImageMemoryBarriers = &barrier;
                dep.memoryBarrierCount = 0;
                dep.pNext = nullptr;
                dep.dependencyFlags = 0;

                for(uint32_t i = 1 ; i < generalImageData.mipMapCount ; i++){
                    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    barrier.subresourceRange.baseMipLevel = i - 1;
                    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

                    vkCmdPipelineBarrier2(cmd , &dep);

                    VkImageBlit blit{};
                    blit.srcOffsets[0] = {0 , 0 ,0 };
                    blit.srcOffsets[1] = {width , height ,1};
                    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    blit.srcSubresource.layerCount = 1;
                    blit.srcSubresource.baseArrayLayer = 0;
                    blit.srcSubresource.mipLevel = i - 1;

                    blit.dstOffsets[0] = {0 , 0 ,0 };
                    blit.dstOffsets[1] = {(width  > 1 ?  width /2 : 1) , (height  > 1 ? height /2 : 1) ,1};
                    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    blit.dstSubresource.layerCount = 1;
                    blit.dstSubresource.baseArrayLayer = 0;
                    blit.dstSubresource.mipLevel = i;

                    vkCmdBlitImage(cmd , image , VK_IMAGE_LAYOUT_GENERAL , image , VK_IMAGE_LAYOUT_GENERAL , 1 , &blit , VK_FILTER_LINEAR);

                    if(width > 1) width /= 2;
                    if(height > 1) height /= 2;

                    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

                    vkCmdPipelineBarrier2(cmd , &dep);

                }

                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.subresourceRange.baseMipLevel = generalImageData.mipMapCount - 1;

                vkCmdPipelineBarrier2(cmd , &dep);
            };

            context->queueManager->Begin();

            context->queueManager->recordCmd(recordBuffer , VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            
            auto result = context->queueManager->End();
            if(!result.has_value()){
                return std::unexpected(result.error());
            }

            return {};
        }


        [[nodiscard]] inline ErrorDataOutput<void> copyBufferToImage(VkCommandBuffer &cmd , VkBuffer &src , VkImage &dst ,const VkDeviceSize size , VkExtent2D extent) noexcept {
            if(!src || !dst || size <= 0 || !cmd){
                const ErrorDataType error{MOX_ERROR_TYPE_WARNING , " in order to copy buffer to image , size has to be > 0 and valid variables"};
            }

            VkBufferImageCopy copy{};
            copy.bufferImageHeight= 0 ;
            copy.bufferRowLength = 0;
            copy.bufferOffset = 0;
            copy.imageExtent = {extent.width, extent.height , 1};
            copy.imageOffset ={0, 0 ,0 };
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.mipLevel = 0;
            copy.imageSubresource.layerCount = 1;

            vkCmdCopyBufferToImage(cmd ,src , dst , VK_IMAGE_LAYOUT_GENERAL , 1 , &copy);

            return {};
        }
    };

}