#pragma once

#include "RenderImage2.hpp"
#include "Skybox2.hpp"

namespace mox{

    // FrameTargets is the set of typical images that are required among many passes
    class FrameTargets final{
    public:
        FrameTargets() = delete;
        FrameTargets(FrameTargets& input) = delete;
        FrameTargets(const FrameTargets& input) = delete;
        FrameTargets& operator=(FrameTargets& input) = delete;
        FrameTargets& operator=(const FrameTargets& input) = delete;

        FrameTargets(FrameTargets&& input){
            processCopy(std::move(input));
        }
        FrameTargets& operator=(FrameTargets&& input) noexcept{
            processCopy(std::move(input));
            return *this;
        }

        explicit FrameTargets(VulkanContext* context){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice , context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }

            const auto result = createImages(context);
            if(!result.has_value()){
                engineLogger(result.error());
                processDelete();
                THROW_MESSAGE;
            }

            this->context = context;
        }

        pRenderImage2 positionImage[2]{};
        pRenderImage2 depthImage[2]{};
        pRenderImage2 normalImage[2]{};
        pRenderImage2 diffuseImage[2]{};
        pRenderImage2 specularImage[2]{};
        // 10
        pRenderImage2 indirectImage[2]{};
        // 13
        pRenderImage2 roughnessImage{};
        pRenderImage2 metallicImage{};
        pRenderImage2 albedoImage{};
        // 16
        
        pRenderImage2 shadowImage[2]{};
        pRenderImage2 ambientOcclusionImage[2]{};
        // 20
        pRenderImage2 filteredShadowImage{};
        pRenderImage2 filteredAmbientOcclusion{};
        // 22
        pRenderImage2 HDRImage[2]{};
        pRenderImage2 finalImage{nullptr};
        // 25
        // the fragment shader of the deferred render marks the pixels that it drew , the HDR pass reads them
        pRenderImage2 pixelCheck{nullptr};
        pRenderImage2 motionVector{nullptr};
        pRenderImage2 viewImage{nullptr};

        pRenderImage2 fogImage;
        pRenderImage2 transmissionLUT;
        pRenderImage2 multiScatterLUT;
        pRenderImage2 multiScatter;
        pRenderImage2 SkyView;
        pSkybox2 skybox;

        ~FrameTargets() noexcept{
            processDelete();
        }

        // cleans the images of the frame : the ones that are written in this frame ( the current half of every ping-pong pair and the
        // images that have no pair ) . the other half is the previous frame , it is the history of the temporal passes and has to stay
        void cleanImages(VkCommandBuffer &cmd) noexcept{
            if(!context) return;

            // the same index that every pass takes
            const uint8_t currentImage = context->sc->frameIndex % 2;

            // the old contents are not needed : the transition from UNDEFINED , and the clear after it
            translateAllImages(cmd , currentImage , VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT , true);
            fillImages(cmd , currentImage);
            // the clear has to stay : from here it is an ordinary transition
            translateAllImages(cmd , currentImage , VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT , false);
        }

    private:

        VulkanContext* context{nullptr};

        void translateAllImages(VkCommandBuffer &cmd , const uint8_t id , const VkAccessFlags2 access , const VkPipelineStageFlags2 stage , const bool fromUndefined) noexcept{
            const auto translate = [&](RenderImage2* image){
                return fromUndefined ? image->translateImageUndefined(access , stage) : image->translateImage(access , stage);
            };

            std::array<VkImageMemoryBarrier2 , 17> barriers;
            barriers[0] = translate(positionImage[id].get());
            barriers[1] = translate(normalImage[id].get());
            barriers[2] = translate(depthImage[id].get());
            barriers[3] = translate(diffuseImage[id].get());
            barriers[4] = translate(specularImage[id].get());
            barriers[5] = translate(indirectImage[id].get());
            barriers[6] = translate(shadowImage[id].get());
            barriers[7] = translate(ambientOcclusionImage[id].get());
            barriers[8] = translate(HDRImage[id].get());

            barriers[9] = translate(roughnessImage.get());
            barriers[10] = translate(metallicImage.get());
            barriers[11] = translate(albedoImage.get());
            barriers[12] = translate(filteredShadowImage.get());
            barriers[13] = translate(filteredAmbientOcclusion.get());
            barriers[14] = translate(finalImage.get());
            barriers[15] = translate(motionVector.get());
            barriers[16] = translate(viewImage.get());

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 0;
            info.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            info.pMemoryBarriers = nullptr;
            info.pImageMemoryBarriers = barriers.data();
            info.pBufferMemoryBarriers = nullptr;
            info.memoryBarrierCount = 0;
            info.pNext = nullptr;
            info.dependencyFlags = 0;

            vkCmdPipelineBarrier2(cmd , &info);
        }


        void fillImages(VkCommandBuffer& cmd , const uint8_t id) noexcept{
            positionImage[id]->cleanTheImage(cmd);
            depthImage[id]->cleanTheImage(cmd);
            diffuseImage[id]->cleanTheImage(cmd);
            shadowImage[id]->cleanTheImage(cmd);
            ambientOcclusionImage[id]->cleanTheImage(cmd);
            normalImage[id]->cleanTheImage(cmd);
            specularImage[id]->cleanTheImage(cmd);
            HDRImage[id]->cleanTheImage(cmd);
            indirectImage[id]->cleanTheImage(cmd);

            albedoImage->cleanTheImage(cmd);
            filteredShadowImage->cleanTheImage(cmd);
            filteredAmbientOcclusion->cleanTheImage(cmd);
            roughnessImage->cleanTheImage(cmd);
            metallicImage->cleanTheImage(cmd);
            finalImage->cleanTheImage(cmd);
            motionVector->cleanTheImage(cmd);
            viewImage->cleanTheImage(cmd);
        }

        [[nodiscard]] ErrorDataOutput<void> createImages(VulkanContext* context) noexcept{
            try{
                RenderImage2::CreateInfo info{};
                info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                info.imageType = VK_IMAGE_TYPE_2D;
                info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                info.context = context;
                info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                info.extent = context->sc->extent;
                info.mipMapCount = 1;
                info.layerCount = 1;
                info.clear.float32[0] = 0.0f;
                info.clear.float32[1] = 0.0f;
                info.clear.float32[2] = 0.0f;
                info.clear.float32[3] = 0.0f;

                // position
                info.imageFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
                positionImage[0] = std::make_shared<RenderImage2>(info);
                positionImage[1] = std::make_shared<RenderImage2>(info);
                viewImage = std::make_shared<RenderImage2>(info);

                // normal
                info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                normalImage[1] = std::make_shared<RenderImage2>(info);
                normalImage[0] = std::make_shared<RenderImage2>(info);

                albedoImage = std::make_shared<RenderImage2>(info);

                indirectImage[0] = std::make_shared<RenderImage2>(info);
                indirectImage[1] = std::make_shared<RenderImage2>(info);

                diffuseImage[0] = std::make_shared<RenderImage2>(info);
                diffuseImage[1] = std::make_shared<RenderImage2>(info);

                specularImage[0] = std::make_shared<RenderImage2>(info);
                specularImage[1] = std::make_shared<RenderImage2>(info);

                HDRImage[0] = std::make_shared<RenderImage2>(info);
                HDRImage[1] = std::make_shared<RenderImage2>(info);

                info.imageFormat = VK_FORMAT_R8G8B8A8_UNORM;
                finalImage = std::make_shared<RenderImage2>(info);

                info.imageFormat = VK_FORMAT_R16G16_SFLOAT;
                motionVector = std::make_shared<RenderImage2>(info);
                
                // depth
                info.imageFormat = VK_FORMAT_R32_SFLOAT;
                depthImage[0]=  std::make_shared<RenderImage2>(info);
                depthImage[1]=  std::make_shared<RenderImage2>(info);

                roughnessImage=  std::make_shared<RenderImage2>(info);
                metallicImage=  std::make_shared<RenderImage2>(info);
                pixelCheck=  std::make_shared<RenderImage2>(info);

                info.clear.float32[0] = 1.0f;
                shadowImage[0] =  std::make_shared<RenderImage2>(info);
                shadowImage[1] =  std::make_shared<RenderImage2>(info);

                ambientOcclusionImage[0] =  std::make_shared<RenderImage2>(info);
                ambientOcclusionImage[1] =  std::make_shared<RenderImage2>(info);

                filteredAmbientOcclusion=  std::make_shared<RenderImage2>(info);
                filteredShadowImage=  std::make_shared<RenderImage2>(info);

                {
                    RenderImage2::CreateInfo info{};
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.context = context;
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    info.extent = {192, 108};
                    info.mipMapCount = 1;
                    info.layerCount = 1;
                    info.clear.float32[0] = 0.0f;
                    info.clear.float32[1] = 0.0f;
                    info.clear.float32[2] = 0.0f;
                    info.clear.float32[3] = 0.0f;
                    info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

                    transmissionLUT = std::make_shared<RenderImage2>(info);
                    multiScatterLUT = std::make_shared<RenderImage2>(info);
                    SkyView = std::make_shared<RenderImage2>(info);

                    // the froxel volume : 160 x 90 pixels and 64 layers ( the dispatch of the VolumentricRender covers exactly this )
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
                    info.layerCount = 64;
                    info.extent = {160 , 90};

                    fogImage = std::make_shared<RenderImage2>(info);
                }

                {
                    RenderImage2::CreateInfo info{};
                    info.viewType = VK_IMAGE_VIEW_TYPE_1D;
                    info.imageType = VK_IMAGE_TYPE_1D;
                    info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.context = context;
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    info.extent = {512 , 1};
                    info.mipMapCount = 1;
                    info.layerCount = 1;
                    info.clear.float32[0] = 0.0f;
                    info.clear.float32[1] = 0.0f;
                    info.clear.float32[2] = 0.0f;
                    info.clear.float32[3] = 0.0f;
                    info.imageFormat = VK_FORMAT_R32_SFLOAT;

                    multiScatter = std::make_shared<RenderImage2>(info);
                }

                {
                    // skybox
                    Skybox2::CreateInfo info{};
                    info.context = context;
                    info.extent = {128 , 128};
                    info.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
                    info.isTexture = false;
                    info.resolution = 128;
                    info.usage =  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

                    skybox = std::make_shared<Skybox2>(info);
                }
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create typical images"});
            }

            return {};
        }

        void processCopy(FrameTargets&& input) noexcept {
            // we can check only one image , since they are all similar
            if(albedoImage == input.albedoImage) return;
            processDelete();
            context = std::exchange(input.context , nullptr);
            std::move(std::begin(input.positionImage), std::end(input.positionImage), std::begin(positionImage));
            std::move(std::begin(input.depthImage),    std::end(input.depthImage),    std::begin(depthImage));
            std::move(std::begin(input.normalImage),   std::end(input.normalImage),   std::begin(normalImage));
            std::move(std::begin(input.diffuseImage),  std::end(input.diffuseImage),  std::begin(diffuseImage));
            std::move(std::begin(input.specularImage), std::end(input.specularImage), std::begin(specularImage));

            std::move(std::begin(input.indirectImage), std::end(input.indirectImage), std::begin(indirectImage));

            roughnessImage = std::move(input.roughnessImage);
            metallicImage  = std::move(input.metallicImage);
            albedoImage    = std::move(input.albedoImage);
            finalImage     = std::move(input.finalImage);
            pixelCheck     = std::move(input.pixelCheck);
            motionVector = std::move(input.motionVector);

            std::move(std::begin(input.shadowImage),           std::end(input.shadowImage),           std::begin(shadowImage));
            std::move(std::begin(input.ambientOcclusionImage), std::end(input.ambientOcclusionImage), std::begin(ambientOcclusionImage));
            filteredShadowImage      = std::move(input.filteredShadowImage);
            filteredAmbientOcclusion = std::move(input.filteredAmbientOcclusion);

            std::move(std::begin(input.HDRImage), std::end(input.HDRImage), std::begin(HDRImage));

            viewImage = std::move(input.viewImage);

            fogImage = std::move(input.fogImage);
            transmissionLUT = std::move(input.transmissionLUT);
            multiScatterLUT = std::move(input.multiScatterLUT);
            multiScatter = std::move(input.multiScatter);
            SkyView = std::move(input.SkyView);
            skybox = std::move(input.skybox);
        }

        void processDelete() noexcept {
            context = nullptr;
            std::fill(std::begin(positionImage), std::end(positionImage), nullptr);
            std::fill(std::begin(depthImage),    std::end(depthImage),    nullptr);
            std::fill(std::begin(normalImage),   std::end(normalImage),   nullptr);
            std::fill(std::begin(diffuseImage),  std::end(diffuseImage),  nullptr);
            std::fill(std::begin(specularImage), std::end(specularImage), nullptr);

            std::fill(std::begin(indirectImage), std::end(indirectImage), nullptr);

            roughnessImage = nullptr;
            metallicImage  = nullptr;
            albedoImage    = nullptr;
            finalImage     = nullptr;
            pixelCheck     = nullptr;
            motionVector = nullptr;

            std::fill(std::begin(shadowImage),           std::end(shadowImage),           nullptr);
            std::fill(std::begin(ambientOcclusionImage), std::end(ambientOcclusionImage), nullptr);
            filteredShadowImage      = nullptr;
            filteredAmbientOcclusion = nullptr;

            std::fill(std::begin(HDRImage), std::end(HDRImage), nullptr);

            viewImage = nullptr;

            fogImage = nullptr;
            transmissionLUT = nullptr;
            multiScatterLUT = nullptr;
            multiScatter = nullptr;
            SkyView = nullptr;
            skybox = nullptr;
        }
    };

    using pFrameTargets = std::shared_ptr<FrameTargets>;
}