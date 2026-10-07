#pragma once

#include "ImageBase.hpp"
#include "../Buffers/buffer.hpp"

namespace mox{

    class Skybox;
    using pSkybox = std::shared_ptr<Skybox>;

    class Skybox final : public ImageBase{
    public:

        Skybox() = delete;
        Skybox(Skybox& input) = delete;
        Skybox& operator=(Skybox& input) = delete;
        Skybox& operator=(Skybox&& input) noexcept{
            performCopy(std::move(input));
            view = std::exchange(input.view , nullptr);
            return *this;
        }
        Skybox(Skybox&& input) noexcept : ImageBase(std::move(input)){
            view = std::exchange(input.view , nullptr);
        }

        ~Skybox(){
            if(view && device) vkDestroyImageView(device , view , nullptr);
            view = nullptr;
        }

        struct CreateInfo{

            VulkanContext* context;

            std::string path{};
            VkImageUsageFlags usage;
            // this is for nonTexture
            VkImageLayout finalLayout;
            VkExtent2D extent;
            // if its a texture
            uint8_t isTexture = false;
            
            uint32_t resolution;
        };

        Skybox(const CreateInfo &info) : ImageBase(info.context){
            if(!info.context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create skybox , since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }
            if(!info.context->logicalDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create skybox , since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(!info.context->queueManager){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create skybox , since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }   

            device = context->logicalDevice->device;

            const auto result = createSkybox(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                // the destructor of Skybox never runs when the constructor throws
                if(view) vkDestroyImageView(device , view , nullptr);
                view = nullptr;
                THROW_MESSAGE;
            }
        }

        VkImageView view = nullptr;
    private:

        [[nodiscard]] ErrorDataOutput<void> createSkybox(const CreateInfo &data){
            if(!data.context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create skybox , since the context has null data"};
                return std::unexpected(error);
            }
            if(!data.context->logicalDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create skybox , since the context has null data"};
                return std::unexpected(error);
            }

            if(!data.context->queueManager){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create skybox , since the context has null data"};
                return std::unexpected(error);
            }   

            if(data.isTexture){
                const auto result = processTextureSkybox(data);
                if(!result.has_value()){
                    return std::unexpected(result.error());
                }
            }else{
                const auto result = processNonTextureSkybox(data);
                if(!result.has_value()){
                    return std::unexpected(result.error());
                }
            }

            return {};
        }

        // the source is a horizontal cross (4 x 3 faces) , imageWidth is the width of the whole cross in pixels
        void copySkybox( VkCommandBuffer &cmd , VkImage &dst , VkBuffer &src ,const  uint32_t &faceSize , const uint32_t imageWidth){
            struct FaceOffset { uint32_t col, row; };
            const FaceOffset faceLayout[6] = {
                {2, 1},   // +X (right)
                {0, 1},   // -X (left)
                {1, 0},   // +Y (top)
                {1, 2},   // -Y (bottom)
                {1, 1},   // +Z (front)
                {3, 1},   // -Z (back)
            };

            for(uint32_t face = 0; face < 6; face++){
                VkBufferImageCopy region{};
                // rows of the source are as wide as the whole cross
                region.bufferRowLength = imageWidth;
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = face;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0 , 0 , 0};
                region.imageExtent = {faceSize, faceSize, 1};

                const VkDeviceSize srcX = static_cast<VkDeviceSize>(faceLayout[face].col) * faceSize;
                const VkDeviceSize srcY = static_cast<VkDeviceSize>(faceLayout[face].row) * faceSize;
                // offset in bytes to the first texel of the face
                region.bufferOffset = (srcY * imageWidth + srcX) * 4;

                vkCmdCopyBufferToImage(cmd, src, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            }
        }

        [[nodiscard]] ErrorDataOutput<void> processTextureSkybox(const CreateInfo &data) noexcept{

            if(data.path.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create skybox , since the paths are nullptr"});
            }

            int width , height;
            unsigned char* text = stbi_load(data.path.c_str() , &width , &height , 0 , 4);
            if(!text){
                stbi_image_free(text);
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the skybox -> {}" , data.path)});
            }

            
            std::vector<uint8_t> buffer(text , text + ((width * height) * 4));
            stbi_image_free(text);

            // a horizontal cross has to be 4 faces wide and 3 faces high
            if(static_cast<uint32_t>(width) != data.resolution * 4 || static_cast<uint32_t>(height) != data.resolution * 3){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the skybox -> {} , the image is {}x{} , but a cross with resolution {} has to be {}x{}" , data.path , width , height , data.resolution , data.resolution * 4 , data.resolution * 3)});
            }

            {
                ImageInfo info{};
                info.imageFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
                info.imageType = VK_IMAGE_TYPE_2D;
                // the picture is sRGB encoded : with the SRGB format the sampler returns linear values , the lighting works in linear
                // ( it was UNORM : the sky went through the gamma of HDR_final a second time and came out washed out )
                info.imageFormat = VK_FORMAT_R8G8B8A8_SRGB;
                info.imageUsage = data.usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
                info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                info.context = context;
                info.extent = {data.resolution, data.resolution};
                info.layerCount = 6;
                info.mipMapCount = 1;

                {
                    const auto result = createImage(info);
                    if(!result.has_value()){
                        return std::unexpected(result.error());
                    }
                }

                {
                    const auto result = createImageView(view , 6 , 0 , 1 , 0);
                    if(!result.has_value()){
                        return std::unexpected(result.error());
                    }
                }
            }

            HostBuffer::CreateInfo info{};
            info.context = context;
            info.size = buffer.size();
            info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

            try{
                HostBuffer staging(info, buffer);

                cmdFunctionType recordBuffer = [&](VkCommandBuffer &cmd) -> void{
                    if(!cmd) return;

                    auto barrier =  translateUnified(VK_IMAGE_LAYOUT_UNDEFINED , VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL , 6 , 0 , 1 , 0);

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

                    copySkybox(cmd , image , staging.buffer , data.resolution , static_cast<uint32_t>(width));

                    barrier =  translateUnified(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL , VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL , 6 , 0 , 1 , 0);
                    dep.pImageMemoryBarriers = &barrier;
                    vkCmdPipelineBarrier2(cmd , &dep);
                };

                context->queueManager->Begin();

                context->queueManager->recordCmd(recordBuffer , VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

                const auto result = context->queueManager->End();

                if(!result.has_value()){
                    return std::unexpected(result.error());
                }

            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "probably failed to create the buffer , since thats the only reason in processTextureSkybox"});
            }


            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> processNonTextureSkybox(const CreateInfo &data) noexcept{

            ImageInfo info{};
            info.imageFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
            info.imageType = VK_IMAGE_TYPE_2D;
            // IBL writes linear irradiance into it , it can be above 1 and it is dark for a night sky : 8 bit UNORM clamped it and banded it
            info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
            info.imageUsage = data.usage;
            info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
            info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
            info.context = context;
            info.extent = {data.resolution, data.resolution};
            info.layerCount = 6;
            info.mipMapCount = 1;

            {
                const auto result = createImage(info);
                if(!result.has_value()){
                    return std::unexpected(result.error());
                }
            }

            {
                const auto result = createImageView(view , 6 , 0 , 1 , 0);
                if(!result.has_value()){
                    return std::unexpected(result.error());
                }
            }

            // now we have to translate the cube
            cmdFunctionType recordBuffer = [&](VkCommandBuffer &cmd) -> void{
                if(!cmd) return;

                auto barrier =  translateUnified(VK_IMAGE_LAYOUT_UNDEFINED , data.finalLayout , 6 , 0 , 1 , 0);

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
            };

            context->queueManager->Begin();

            context->queueManager->recordCmd(recordBuffer , VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

            const auto result = context->queueManager->End();

            if(!result.has_value()){
                return std::unexpected(result.error());
            }

            return {};
        }

    };

}