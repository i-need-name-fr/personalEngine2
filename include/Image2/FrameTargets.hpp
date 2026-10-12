#pragma once

#include "RenderImage2.hpp"

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
        }

        pRenderImage2 positionImage[2]{};
        pRenderImage2 depthImage[2]{};
        pRenderImage2 normalImage[2]{};
        pRenderImage2 diffuseImage[2]{};
        pRenderImage2 specularImage[2]{};
        // 10
        pRenderImage2 indirectImage[2]{};
        pRenderImage2 ambientImage{};
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

        ~FrameTargets() noexcept{
            processDelete();
        }

        void cleanImages(VkCommandBuffer &cmd) noexcept{
            // we need to clean
            translateAllImages(cmd , VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
            fillImages(cmd);
            translateAllImages(cmd , VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        }

    private:

        void translateAllImages(VkCommandBuffer &cmd, const VkAccessFlags2 access ,const VkPipelineStageFlags2 stage) noexcept{
            std::array<VkImageMemoryBarrier2, 25> barriers;
            for(auto& id : { 0 ,1}){
                barriers[0 + id] = positionImage[id]->translateImageUndefined(access , stage);
                barriers[2 + id] = normalImage[id]->translateImageUndefined(access , stage);
                barriers[4 + id] = depthImage[id]->translateImageUndefined(access , stage);
                barriers[6 + id] = diffuseImage[id]->translateImageUndefined(access , stage);
                barriers[8 + id] = specularImage[id]->translateImageUndefined(access , stage);
                barriers[10 + id] = indirectImage[id]->translateImageUndefined(access , stage);
                barriers[12 + id] = shadowImage[id]->translateImageUndefined(access , stage);
                barriers[14 + id] = ambientOcclusionImage[id]->translateImageUndefined(access , stage);
                barriers[16 + id] = HDRImage[id]->translateImageUndefined(access , stage);
            }
            barriers[18] = ambientImage->translateImageUndefined(access , stage);
            barriers[19] = roughnessImage->translateImageUndefined(access , stage);
            barriers[20] = metallicImage->translateImageUndefined(access , stage);
            barriers[21] = albedoImage->translateImageUndefined(access , stage);
            barriers[22] = filteredShadowImage->translateImageUndefined(access , stage);
            barriers[23] = filteredAmbientOcclusion->translateImageUndefined(access , stage);
            barriers[24] = finalImage->translateImageUndefined(access , stage);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 0;
            info.imageMemoryBarrierCount = 25;
            info.pMemoryBarriers = nullptr;
            info.pImageMemoryBarriers = barriers.data();
            info.pBufferMemoryBarriers = nullptr;
            info.memoryBarrierCount = 0;
            info.pNext = nullptr;
            info.dependencyFlags = 0;

            vkCmdPipelineBarrier2(cmd , &info);
        }


        void fillImages(VkCommandBuffer& cmd) noexcept{
            for(auto& id : { 0 ,1}){
                positionImage[id]->cleanTheImage(cmd);
                depthImage[id]->cleanTheImage(cmd);
                diffuseImage[id]->cleanTheImage(cmd);
                shadowImage[id]->cleanTheImage(cmd);
                ambientOcclusionImage[id]->cleanTheImage(cmd);
                normalImage[id]->cleanTheImage(cmd);
                specularImage[id]->cleanTheImage(cmd);
                HDRImage[id]->cleanTheImage(cmd);
                indirectImage[id]->cleanTheImage(cmd);
            }

            albedoImage->cleanTheImage(cmd);
            filteredShadowImage->cleanTheImage(cmd);
            filteredAmbientOcclusion->cleanTheImage(cmd);
            roughnessImage->cleanTheImage(cmd);
            metallicImage->cleanTheImage(cmd);
            finalImage->cleanTheImage(cmd);
            ambientImage->cleanTheImage(cmd);
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

                // normal
                info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                normalImage[1] = std::make_shared<RenderImage2>(info);
                normalImage[0] = std::make_shared<RenderImage2>(info);

                ambientImage = std::make_shared<RenderImage2>(info);
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
            std::move(std::begin(input.positionImage), std::end(input.positionImage), std::begin(positionImage));
            std::move(std::begin(input.depthImage),    std::end(input.depthImage),    std::begin(depthImage));
            std::move(std::begin(input.normalImage),   std::end(input.normalImage),   std::begin(normalImage));
            std::move(std::begin(input.diffuseImage),  std::end(input.diffuseImage),  std::begin(diffuseImage));
            std::move(std::begin(input.specularImage), std::end(input.specularImage), std::begin(specularImage));

            std::move(std::begin(input.indirectImage), std::end(input.indirectImage), std::begin(indirectImage));
            ambientImage = std::move(input.ambientImage);

            roughnessImage = std::move(input.roughnessImage);
            metallicImage  = std::move(input.metallicImage);
            albedoImage    = std::move(input.albedoImage);
            finalImage     = std::move(input.finalImage);
            pixelCheck     = std::move(input.pixelCheck);

            std::move(std::begin(input.shadowImage),           std::end(input.shadowImage),           std::begin(shadowImage));
            std::move(std::begin(input.ambientOcclusionImage), std::end(input.ambientOcclusionImage), std::begin(ambientOcclusionImage));
            filteredShadowImage      = std::move(input.filteredShadowImage);
            filteredAmbientOcclusion = std::move(input.filteredAmbientOcclusion);

            std::move(std::begin(input.HDRImage), std::end(input.HDRImage), std::begin(HDRImage));
        }

        void processDelete() noexcept {
            std::fill(std::begin(positionImage), std::end(positionImage), nullptr);
            std::fill(std::begin(depthImage),    std::end(depthImage),    nullptr);
            std::fill(std::begin(normalImage),   std::end(normalImage),   nullptr);
            std::fill(std::begin(diffuseImage),  std::end(diffuseImage),  nullptr);
            std::fill(std::begin(specularImage), std::end(specularImage), nullptr);

            std::fill(std::begin(indirectImage), std::end(indirectImage), nullptr);
            ambientImage = nullptr;

            roughnessImage = nullptr;
            metallicImage  = nullptr;
            albedoImage    = nullptr;
            finalImage     = nullptr;
            pixelCheck     = nullptr;

            std::fill(std::begin(shadowImage),           std::end(shadowImage),           nullptr);
            std::fill(std::begin(ambientOcclusionImage), std::end(ambientOcclusionImage), nullptr);
            filteredShadowImage      = nullptr;
            filteredAmbientOcclusion = nullptr;

            std::fill(std::begin(HDRImage), std::end(HDRImage), nullptr);
        }
    };

    using pFrameTargets = std::shared_ptr<FrameTargets>;
}