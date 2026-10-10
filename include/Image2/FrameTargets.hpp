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

        pRenderImage2 indirectImage[2]{};
        pRenderImage2 ambientImage{};

        pRenderImage2 roughnessImage{};
        pRenderImage2 metallicImage{};
        pRenderImage2 albedoImage{};

        
        pRenderImage2 shadowImage[2]{};
        pRenderImage2 ambientOcclusionImage[2]{};
        pRenderImage2 filteredShadowImage{};
        pRenderImage2 filteredAmbientOcclusion{};

        pRenderImage2 HDRImage[2]{};

        ~FrameTargets() noexcept{
            processDelete();
        }

    private:

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

                // depth
                info.imageFormat = VK_FORMAT_R32_SFLOAT;
                depthImage[0]=  std::make_shared<RenderImage2>(info);
                depthImage[1]=  std::make_shared<RenderImage2>(info);

                roughnessImage=  std::make_shared<RenderImage2>(info);
                metallicImage=  std::make_shared<RenderImage2>(info);

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

            std::fill(std::begin(shadowImage),           std::end(shadowImage),           nullptr);
            std::fill(std::begin(ambientOcclusionImage), std::end(ambientOcclusionImage), nullptr);
            filteredShadowImage      = nullptr;
            filteredAmbientOcclusion = nullptr;

            std::fill(std::begin(HDRImage), std::end(HDRImage), nullptr);
        }
    };

    using pFrameTargets = std::shared_ptr<FrameTargets>;
}