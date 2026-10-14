#pragma once

#include "GeneralBuffer.hpp"


namespace mox{

    class SourceBuffers final{
    public:

        SetTypicalOperators(SourceBuffers)
        // both are default , since we got the shared ptrs
        SourceBuffers& operator=(SourceBuffers&& input) = default;
        SourceBuffers(SourceBuffers&& input) = default;

        explicit SourceBuffers(VulkanContext* context){
            const auto result = createAllBuffers(context);
            if(!result.has_value()){
                processDelete();
                engineLogger(result);
                THROW_MESSAGE;
            }
        }

        pGeneralBuffer2<BufferType::Device , true> generalBuffer{nullptr};
        pGeneralBuffer2<BufferType::Device , true> features{nullptr};
        pGeneralBuffer2<BufferType::Device , true> sunBuffer{nullptr};
        pGeneralBuffer2<BufferType::Device , true> lightBuffer{nullptr};
        pGeneralBuffer2<BufferType::Device , true> luminanceHistory{nullptr};

        pGeneralBuffer2<BufferType::Device , true> forwardInfo{nullptr};
        // since we dont really care about reading it 

        pGeneralBuffer2<BufferType::Device , false> tileLightIndices{nullptr};
        pGeneralBuffer2<BufferType::Device , false> lightCountPerTile{nullptr};

        ~SourceBuffers(){
            processDelete();
        }

    private:

        void processDelete() noexcept{
            generalBuffer = nullptr;
            features = nullptr;
            sunBuffer = nullptr;
            lightBuffer = nullptr;
            luminanceHistory = nullptr;
            
            tileLightIndices = nullptr;
            lightCountPerTile = nullptr;
            forwardInfo = nullptr;
        }

        [[nodiscard]] ErrorDataOutput<void> createAllBuffers(VulkanContext* context) noexcept{

            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeInstance | VkReqTypePhysicalDevice , context);
                if(!result.has_value()) return result;
            }
            try{

                GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                info.context= context;
                info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                
                // general buffer
                info.size = sizeof(GeneralInfo);
                generalBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                // features
                info.size = sizeof(GeneralFeatures);
                features = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                // sun buffer
                info.size = sizeof(SunSource);
                sunBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                // light buffer
                info.size = sizeof(LightSource) * MAX_LIGHT_SOURCE;
                lightBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                
                // luminance history
                info.size = sizeof(uint32_t) * 257;
                luminanceHistory = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                ForwardPlusInfo data{};
                data.TILE_SIZE = 16;

                uint32_t tilesX = (context->sc->extent.width  + data.TILE_SIZE - 1) / data.TILE_SIZE;
                uint32_t tilesY = (context->sc->extent.height + data.TILE_SIZE - 1) / data.TILE_SIZE;

                data.TILE_PER_ROW      = tilesX;
                data.TILE_TOTAL_AMOUNT = tilesX * tilesY;
                data.totalLights       = 0;

                info.size = sizeof(ForwardPlusInfo);
                forwardInfo = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info , &data);

                {   


                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context= context;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.size = sizeof(uint32_t) * MAX_LIGHT_SOURCE * data.TILE_TOTAL_AMOUNT;

                    tileLightIndices = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info);

                    info.size = sizeof(uint32_t) * data.TILE_TOTAL_AMOUNT;

                    lightCountPerTile = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION, "failed to create typical source buffers"});
            }

            return {};
        }
    };
}
