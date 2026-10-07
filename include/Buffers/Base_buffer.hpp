#pragma once 

#include "../vulkanContext/vulkanBase.hpp"

namespace mox{

    enum class BufferType {
        Device,
        Host,
        Undefined
    };

    class BufferBase;
    using pBufferBase = std::shared_ptr<BufferBase>;

    class BufferBase{
    public:

        struct BufferBaseCreateInfo{
            VulkanContext* context;
            VkDeviceSize size;
            VkBufferUsageFlags usage;
            VkMemoryPropertyFlags memoryProperty;
        };

        BufferBase(BufferBase& input) = delete;
        BufferBase() = delete;
        BufferBase& operator=(BufferBase& input) = delete;

        BufferBase(BufferBase&& input) noexcept{
            performCopy(std::move(input));
        }

        BufferBase& operator=(BufferBase&& input) noexcept{
            performCopy(std::move(input));
            return *this;   
        }

        virtual ~BufferBase(){
            cleanBuffer();
        }

        VkBuffer buffer = nullptr;
        VkDeviceMemory memory = nullptr;
        VkDeviceAddress pBuffer = 0;
        void* data = nullptr;

        BufferBase(VulkanContext* context){
            [[unlikely]] if(!context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            [[unlikely]] if(!context->queueManager){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            [[unlikely]] if(!context->logicalDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            this->context = context;
            this->device = context->logicalDevice->device;
        }

        BufferBase(const BufferBaseCreateInfo &info){
            [[unlikely]] if(!info.context){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            [[unlikely]] if(!info.context->queueManager){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            [[unlikely]] if(!info.context->logicalDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer base since the context has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            context = info.context;
            device = context->logicalDevice->device;


            const auto result = createBuffer(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        VkDeviceSize getBufferSize() const noexcept{
            return bufferSize;
        }

    protected:

        VkDeviceSize bufferSize = 0;
        VulkanContext* context = nullptr;
        VkDevice device = nullptr;
        
        void performCopy(BufferBase&& input) noexcept{
            buffer = std::exchange(input.buffer , nullptr);
            memory = std::exchange(input.memory , nullptr);
            bufferSize = input.bufferSize;
            data = std::exchange(input.data , nullptr);
            pBuffer = std::exchange(input.pBuffer , 0);
            context = std::exchange(input.context , nullptr);
            device = std::exchange(input.device , nullptr);
        }

        void cleanBuffer() noexcept{
            if(device){
                if(buffer) vkDestroyBuffer(device , buffer, nullptr);
                if(memory) vkFreeMemory(device , memory , nullptr);
            }

            device = nullptr;
            buffer = nullptr;
            memory = nullptr;
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
                VkBufferCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                info.flags = 0;
                info.pNext = nullptr;
                info.usage = data.usage;
                info.size = data.size;
                info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                info.queueFamilyIndexCount = 0;

                const auto result = vkCreateBuffer(device , &info , nullptr , &buffer);
                if(result != VK_SUCCESS){
                    const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create buffer , error ID -> {}", static_cast<int32_t>(result))};
                    return std::unexpected(error);
                }

            }

            VkMemoryRequirements req{};

            vkGetBufferMemoryRequirements(device , buffer, &req);

            VkMemoryAllocateFlagsInfo flags{};
            flags.deviceMask = 0;
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            flags.pNext = nullptr;
            flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;

            VkMemoryAllocateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            info.pNext = (data.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT ? &flags : nullptr);
            info.allocationSize = req.size;

            try{
                info.memoryTypeIndex = findMemoryIndex(data.context , req.memoryTypeBits , data.memoryProperty);
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create buffer , since the memory index is undefined"});
            }
            const auto result = vkAllocateMemory(device , &info , nullptr , &memory);

            if(result != VK_SUCCESS){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to allocate memory ,for the buffer, error ID -> {}", static_cast<int32_t>(result))};
                return std::unexpected(error);
            }

            {
                const auto result = vkBindBufferMemory(device , buffer, memory, 0);
                if(result != VK_SUCCESS){
                    const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to bind buffer memory, error ID -> {}", static_cast<int32_t>(result))};
                    return std::unexpected(error);
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
}