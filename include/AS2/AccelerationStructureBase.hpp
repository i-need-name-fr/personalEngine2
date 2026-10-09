#pragma once

#include "../vulkanContext/vulkanBase.hpp"
#include "../Buffer2/GeneralBuffer.hpp"
namespace mox{

    class AccelerationStructure{
    public:

        AccelerationStructure() = delete;
        AccelerationStructure(AccelerationStructure& input) = delete;
        AccelerationStructure(const AccelerationStructure& input) = delete;
        AccelerationStructure(AccelerationStructure&& input) noexcept{
            processCopy(std::move(input));
        }

        AccelerationStructure& operator=(AccelerationStructure& input) = delete;
        AccelerationStructure& operator=(const AccelerationStructure& input) = delete;
        AccelerationStructure& operator=(AccelerationStructure&& input) noexcept{

            processCopy(std::move(input));
            return *this;
        }

        AccelerationStructure(VulkanContext* context){
            // these two used to build the error and then fall through - a null context was
            // dereferenced on the very next line
            if(!context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Acceleration structure , since the context is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager , context);
            if(!result.has_value()){
                engineLogger(result.error());
                THROW_MESSAGE;
            }
            this->device = context->logicalDevice->device;
        }

        virtual ~AccelerationStructure() noexcept{
            processDelete();
        }

        VkAccelerationStructureKHR AS = VK_NULL_HANDLE;
        VkDeviceAddress pAS = 0;
        pGeneralBuffer2<BufferType::Device , false> build = nullptr;;
        pGeneralBuffer2<BufferType::Device , false> scratch = nullptr;

    protected:

        VkDevice device = nullptr;


        void processCopy(AccelerationStructure&& input) noexcept {
            if(input.AS == this->AS) return ;
            processDelete();
            device = std::exchange(input.device , nullptr);
            AS = std::exchange(input.AS , nullptr);
            pAS = std::exchange(input.pAS , 0);
            build = std::move(input.build);
            scratch = std::move(input.scratch);
        }   

        void processDelete() noexcept {

            if(device){
                if(AS) destroyAccelerationStructureKHR(device , AS , nullptr);
            }

            device = nullptr;
            AS = nullptr;
            pAS = 0;
            build = nullptr;
            scratch = nullptr;
        }
    };

}