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
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION, "failed to create typical source buffers"});
            }

            return {};
        }
    };
}
