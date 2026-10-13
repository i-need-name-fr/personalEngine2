#pragma once

#include "../vulkanContext/vulkanBase.hpp"

namespace mox{

    class BufferBase2{
    public:

        struct BufferBaseCreateInfo{
            VulkanContext* context;
            VkDeviceSize size;
            VkBufferUsageFlags usage;
            VkMemoryPropertyFlags memoryProperty;
        };

        BufferBase2(BufferBase2& input) = delete;
        BufferBase2() = delete;
        BufferBase2& operator=(BufferBase2& input) = delete;

        BufferBase2(BufferBase2&& input) noexcept{
            performCopy(std::move(input));
        }

        BufferBase2& operator=(BufferBase2&& input) noexcept{
            if(this == &input) return *this;
            // performCopy only takes the handles of the other buffer , the buffer and the memory of this one would stay alive for ever
            cleanBuffer();
            performCopy(std::move(input));
            return *this;   
        }

        virtual ~BufferBase2(){
            cleanBuffer();
        }

        VkBuffer buffer = nullptr;
        VkDeviceAddress pBuffer = 0;
        void* data = nullptr;

        BufferBase2(VulkanContext* context){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager | VkReqTypePhysicalDevice , context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }            

            this->context = context;
            this->device = context->logicalDevice->device;
        }

        BufferBase2(const BufferBaseCreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager | VkReqTypePhysicalDevice , info.context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }    

            context = info.context;
            device = context->logicalDevice->device;


            const auto result = createBuffer(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                // the destructor does not run when the constructor throws : the buffer that was made before the failure would leak
                cleanBuffer();
                THROW_MESSAGE;
            }
        }

        VkDeviceSize getBufferSize() const noexcept{
            return bufferSize;
        }

        VkBufferMemoryBarrier2 translateBuffer(const VkAccessFlags2 access , const VkPipelineStageFlags2 stage)noexcept{
            VkBufferMemoryBarrier2 barrier{};

            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barrier.buffer = buffer;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.srcAccessMask = currentAccess;
            barrier.srcStageMask = currentStage;
            barrier.dstAccessMask = access;
            barrier.dstStageMask = stage;
            barrier.offset = 0;
            barrier.size = bufferSize;
            barrier.pNext = nullptr;

            currentAccess = access;
            currentStage = stage; 

            return barrier;
        }

    protected:

        VkAccessFlags2 currentAccess{0};
        VkPipelineStageFlags2 currentStage{VK_PIPELINE_STAGE_2_NONE};


        VmaAllocation allocation{nullptr};
        VmaAllocationInfo memoryInfo{};

        VkDeviceSize bufferSize = 0;
        VulkanContext* context = nullptr;
        VkDevice device = nullptr;
        
        void performCopy(BufferBase2&& input) noexcept{
            buffer = std::exchange(input.buffer , nullptr);
            bufferSize = input.bufferSize;
            data = std::exchange(input.data , nullptr);
            pBuffer = std::exchange(input.pBuffer , 0);
            context = std::exchange(input.context , nullptr);
            device = std::exchange(input.device , nullptr);
            currentAccess = input.currentAccess; currentStage= input.currentStage;
            memoryInfo = input.memoryInfo;
            allocation = std::exchange(input.allocation , nullptr);
        }

        void cleanBuffer() noexcept{
            if(device){
                vmaDestroyBuffer(context->allocator->allocation , buffer , allocation);
            }
            allocation = nullptr;
            memoryInfo = {};
            device = nullptr;
            buffer = nullptr;
            bufferSize=  0;
            context = nullptr;
            data= nullptr;
            pBuffer = 0;
        }

        [[nodiscard]] [[gnu::cold]] ErrorDataOutput<void> createBuffer(const BufferBaseCreateInfo &data) noexcept {
            [[unlikely]] if(!data.context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                return std::unexpected(error);
            }

            [[unlikely]] if(!data.context->queueManager){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                return std::unexpected(error);
            }

            [[unlikely]] if(!data.context->logicalDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                return std::unexpected(error);
            }

            {
                bool readable = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & data.memoryProperty) == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                VkBufferCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                info.flags = 0;
                info.pNext = nullptr;
                info.usage = data.usage;
                info.size = data.size;
                info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                info.queueFamilyIndexCount = 0;

                VmaAllocationCreateInfo alloc{};
                alloc.usage = VMA_MEMORY_USAGE_UNKNOWN;
                alloc.requiredFlags = data.memoryProperty;
                if(readable){
                    alloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
                }else{
                    alloc.flags = 0;
                }

                const auto result = vmaCreateBuffer(data.context->allocator->allocation , &info , &alloc , &buffer , &allocation , &memoryInfo);
                if(result != VK_SUCCESS) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create buffer, VMA error ID -> {}", static_cast<int32_t>(result))});

                bufferSize = memoryInfo.size;
                if(readable){
                    this->data = memoryInfo.pMappedData;
                }
            }

            if((data.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)){
                VkBufferDeviceAddressInfo info{};
                info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
                info.pNext=  nullptr;
                info.buffer = buffer;

                pBuffer = vkGetBufferDeviceAddress(device , &info);
                if(!pBuffer){
                    const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create buffer pointer")};
                    return std::unexpected(error);
                }
            }

            bufferSize = data.size;

            return {};
        }

        ErrorDataOutput<void> copyBufferToBuffer(VkBuffer &src , VkBuffer &dst , const VkDeviceSize size) noexcept{
            if(!dst || !src){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED , "failed to copy buffer from src to dst , since the buffers are null"};
                return std::unexpected(error);
            }

            if(size ==0){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED , "failed to copy buffer from src to dst , since the size is 0"};
                return std::unexpected(error);
            }

            cmdFunctionType recordBuffer = [&](VkCommandBuffer &cmd) -> void{
                VkBufferCopy copy{};
                copy.dstOffset = 0;
                copy.srcOffset = 0;
                copy.size = size;

                vkCmdCopyBuffer(cmd  ,src , dst , 1 , &copy);
            };

            context->queueManager->Begin();

            context->queueManager->recordCmd(recordBuffer , VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

            auto result = context->queueManager->End();
            if(!result.has_value()){
                return std::unexpected(result.error());
            }
            return {};
        }
    };

    using pBufferBase2 = std::shared_ptr<BufferBase2>;
}