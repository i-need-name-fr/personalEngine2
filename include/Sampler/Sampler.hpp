#pragma once

#include "../vulkanContext/vulkanBase.hpp"

namespace mox{

    class Sampler;
    using pSampler = std::shared_ptr<Sampler>;

    class Sampler final{
    public:

        Sampler()=  delete;
        Sampler(Sampler& input) = delete;
        Sampler& operator=(Sampler& input) = delete;
        Sampler(Sampler&& input){
            clean();
            device = std::exchange(input.device , nullptr);
            sampler = std::exchange(input.sampler , nullptr);
        }
        Sampler& operator=(Sampler&& input){
            clean();
            device = std::exchange(input.device , nullptr);
            sampler = std::exchange(input.sampler , nullptr);
            return *this;
        }

        ~Sampler(){
            clean();
        }
        

        struct CreateInfo{
            VulkanContext* context = nullptr;

            bool requiredTypicalSampler = false;
            bool textureSampler = false;
            bool borderColorIsWhite = false;
            bool requiredComparingSampler = false;

            VkCompareOp SamplerCompareOp;
        };

        explicit Sampler(const CreateInfo& info){
            if(!info.context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create sampler , since the context is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(!info.context->logicalDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create sampler , since the context logical device  is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            device = info.context->logicalDevice->device;
            auto result = createSampler(info);
            if(result.has_value()){

                const ErrorDataType res = {MOX_ERROR_TYPE_SUCCESS , std::format("successfuly created sampler " )};
                engineLogger(&res);

            }else{
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        VkSampler sampler = nullptr;

    private:

        VkDevice device = nullptr;

        void clean(){
            if(device){
                if(sampler) vkDestroySampler(device , sampler , nullptr);
            }
        }

        [[gnu::cold]] [[nodiscard]] ErrorDataOutput<void> createSampler(const CreateInfo &context) noexcept {

            VkSamplerCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            info.flags = 0;
            info.pNext = nullptr;
            info.maxAnisotropy = 0.0f;
            info.anisotropyEnable = VK_FALSE;

            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            info.unnormalizedCoordinates = VK_FALSE;
            info.mipLodBias = 0.0f;
            info.minLod = 0.0f;
            info.maxLod = 0.0f;

            if(context.requiredTypicalSampler){

                info.minFilter = VK_FILTER_NEAREST;
                info.magFilter = VK_FILTER_NEAREST;     
                // MIRROR_CLAMP_TO_EDGE needs samplerMirrorClampToEdge , which this
                // device was not created with - plain CLAMP_TO_EDGE behaves the same
                // for every lookup this project does
                info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;


            }else if(context.textureSampler){

                info.minFilter = VK_FILTER_LINEAR;
                info.magFilter = VK_FILTER_LINEAR;
                // the textures have a full mip chain ( Textures.hpp builds it ) , but maxLod = 0 above made every lookup read the biggest mip :
                // a minified texture ( the floor in the distance ) aliased and shimmered with every movement of the camera
                info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                info.minLod = 0.0f;
                info.maxLod = VK_LOD_CLAMP_NONE;
                info.anisotropyEnable = VK_TRUE;
                info.maxAnisotropy = 8.0f;
                // plain REPEAT : with MIRRORED_REPEAT every second tile is flipped , and on a tiled surface ( the floor ) the mirror axes
                // show as bright symmetric stripes that run to the horizon
                info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;           
                info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;  
            }

            if(context.borderColorIsWhite){
                info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
            }else{
                info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            }

            if(context.requiredComparingSampler){
                info.compareEnable = VK_TRUE;
                info.compareOp = context.SamplerCompareOp;
            }else{
                info.compareEnable = VK_FALSE;
                info.compareOp = VK_COMPARE_OP_ALWAYS;
            }

            auto result = vkCreateSampler(device , &info , nullptr , &sampler);
            if(result != VK_SUCCESS){
                ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create sampler , error ID -> {}" , static_cast<int32_t>(result))};
                return std::unexpected(error);
            }

            return{};
        }
    };
}