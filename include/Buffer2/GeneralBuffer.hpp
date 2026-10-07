#pragma once

#include "../Buffers/Base_buffer.hpp"
#include <span>


namespace mox{

    // =============================== INFO ===============================
    // new type of Buffer : GeneralBuffer<>
    // this type of buffer can be GPU and CPU , based on type parameter
    // now we dont have to create either host buffer or device buffer
    // this buffer also adapts function based on the input parameters
    // =============================== END ===============================

    template<typename T>
    concept MassiveTypeData = requires(T t){
        {t.size()};
        {t.data()};
    };

    template<typename T>
    concept GPUDataType = !std::is_pointer_v<T> && std::is_standard_layout_v<T> && !MassiveTypeData<T>;

    template<BufferType type , uint8_t readable>
    class GeneralBuffer2 final : BufferBase{
    public:

        GeneralBuffer2() = delete;
        GeneralBuffer2(const GeneralBuffer2& input) = delete;
        GeneralBuffer2(GeneralBuffer2& input) = delete;
        GeneralBuffer2& operator=(GeneralBuffer2& input) = delete;
        GeneralBuffer2& operator=(const GeneralBuffer2& input) = delete;

        GeneralBuffer2(GeneralBuffer2&& input) noexcept : BufferBase(std::move(input)){}

        GeneralBuffer2& operator=(GeneralBuffer2&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }


        struct CreateInfo{
            VulkanContext* context;

            VkDeviceSize size;
            VkBufferUsageFlags usage;
        };

        explicit GeneralBuffer2(const CreateInfo &info) : BufferBase(info.context){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager , info.context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }

            if constexpr(type == BufferType::Host){
                const auto result = createCPUTypeBuffer(info);
                if(!result.has_value()){
                    cleanBuffer();
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }else if constexpr(type == BufferType::Device){
                const auto result = createGPUTypeBuffer(info);
                if(!result.has_value()){
                    cleanBuffer();
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }else{
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create GeneralBuffer , since the buffer type is undefined"};
            }

            if constexpr(readable){
                const auto result = vkMapMemory(info.context->logicalDevice->device , memory , 0 , bufferSize , 0 , &data);
                if(result != VK_SUCCESS){
                    const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create general buffer , since the vk map memory failed , error ID -> {}" , static_cast<int32_t>(result))};
                    engineLogger(error);
                    THROW_MESSAGE;
                }
            }
        }   

        template<typename T>
        requires GPUDataType<T>
        explicit GeneralBuffer2(const CreateInfo &info , std::span<T> inputData ) : GeneralBuffer2(info) {

            try{
                if(!inputData.empty()){
                    VkDeviceSize requiredSize = std::min(inputData.size_bytes() , bufferSize);

                    if constexpr(type == BufferType::Host){
                        std::memcpy(this->data , inputData.data() , requiredSize);
                    }else{

                        // copying data to staging buffer
                        BufferBaseCreateInfo data{};
                        data.context = info.context;
                        data.memoryProperty = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                        data.size = requiredSize;
                        data.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

                        BufferBase staging(data);
                        
                        {
                            const auto result = vkMapMemory(info.context->logicalDevice->device , staging.memory , 0 , requiredSize , 0 , &staging.data);
                            if(result != VK_SUCCESS){
                                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create general buffer , since the vk map memory failed for staging buffer , error ID -> {}" , static_cast<int32_t>(result))};
                                engineLogger(error);
                                THROW_MESSAGE;
                            }

                            std::memcpy(staging.data , inputData.data() , requiredSize );
                        }

                        
                        const auto result = copyBufferToBuffer(staging.buffer , buffer , requiredSize );
                        if(!result.has_value()){
                            engineLogger(result.error());
                            THROW_MESSAGE;
                        }
                    }
                }
            }
            catch(std::exception &e){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create general buffer , check the logs please"};
                engineLogger(error);
                THROW_MESSAGE;
            }
        }

        explicit GeneralBuffer2(const CreateInfo &info , void* inputData ) : GeneralBuffer2(info){
         
            try{

                if constexpr(readable){
                    std::memcpy(this->data , inputData , info.size);
                }else{
                    BufferBaseCreateInfo data{};
                    data.context = info.context;
                    data.memoryProperty = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                    data.size = info.size;
                    data.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

                    BufferBase staging(data);

                    {
                        const auto result = vkMapMemory(info.context->logicalDevice->device , staging.memory , 0 , info.size , 0 , &staging.data);
                        if(result != VK_SUCCESS){
                            const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create general buffer , since the vk map memory failed for staging buffer , error ID -> {}" , static_cast<int32_t>(result))};
                            engineLogger(error);
                            THROW_MESSAGE;
                        }

                        std::memcpy(staging.data , inputData , info.size );
                    }

                    const auto result = copyBufferToBuffer(staging.buffer , buffer , info.size );
                    if(!result.has_value()){
                        engineLogger(result.error());
                        THROW_MESSAGE;
                    }
                }
            }
            catch(std::exception &e){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create general buffer , check the logs please"};
                engineLogger(error);
                THROW_MESSAGE;
            }
        }

        // record <readable>

        template<typename T>
        requires GPUDataType<T>
        void recordData(std::span<T> inputData) noexcept requires(readable == true){
            [[unlikely]] if(inputData.empty()){

                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to record data for general buffer , since the input data is nullptr, message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }

            VkDeviceSize requiredSize = std::min(inputData.size_bytes() , bufferSize);
            std::memcpy(this->data , inputData.data(), requiredSize);
        }

        void recordData(void* inputData , const VkDeviceSize size) noexcept requires(readable == true){
            [[unlikely]] if(!inputData){

                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to record data for general buffer , since the input data is nullptr, message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }

            VkDeviceSize requiredSize = std::min(size , bufferSize);
            std::memcpy(this->data , inputData, requiredSize);
        }

        // get data <readable>

        template<typename T>
        requires GPUDataType<T>
        [[nodiscard]] std::vector<T> getBuffetData(const uint32_t count) const noexcept requires(readable == true) {
            [[unlikely]] if(!count){

                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to get buffer data , since the count for data set to 0, message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }
            VkDeviceSize requiredSize = std::min(bufferSize , sizeof(T) * count);
            std::vector<T> outputBuffer(count);
            std::memcpy(outputBuffer.data() , data , requiredSize);
            return outputBuffer;
        }

        template<typename T>
        requires GPUDataType<T>
        [[nodiscard]] T getBuffetDataSingular() const noexcept requires(readable == true) {
            VkDeviceSize requiredSize = std::min(bufferSize , sizeof(T));
            T output{};
            std::memcpy(&output , data , requiredSize);
            return output;
        }

        // update buffer GPU
        
        void updateBuffer(VkCommandBuffer &cmd) noexcept requires(BufferType::Device == type && !readable){
            [[unlikely]] if(!cmd){

                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to update buffer data , since the command buffer is null, message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }

            vkCmdFillBuffer(cmd , buffer , 0 , bufferSize , 0);
        }

        template<typename T>
        requires GPUDataType<T>
        void updateBuffer(VkCommandBuffer &cmd , std::span<T> &inputData) noexcept requires(BufferType::Device == type && !readable){
            [[unlikely]] if(!cmd){

                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to update buffer data , since the command buffer is null, message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }

            [[unlikely]] if(inputData.empty()){

                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to update bufder , since the input data is empty , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }
            VkDeviceSize requiredSize = std::min(bufferSize , inputData.size_bytes());
            vkCmdUpdateBuffer(cmd , buffer , 0 , requiredSize , inputData.data());
        }

        void updateBuffer(VkCommandBuffer &cmd , void* inputData , const VkDeviceSize size ) noexcept requires(BufferType::Device == type && !readable){
            [[unlikely]] if(!cmd){

                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to update buffer data , since the command buffer is null, message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }

            [[unlikely]] if(!inputData){

                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to update bufder , since the input data is empty , message Count {} " , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }
            VkDeviceSize requiredSize = std::min(bufferSize , size);
            vkCmdUpdateBuffer(cmd , buffer , 0 , requiredSize , inputData);
        }

        ~GeneralBuffer2(){}

    private:

        [[nodiscard]] ErrorDataOutput<void> createCPUTypeBuffer(const CreateInfo &input) noexcept{

            BufferBaseCreateInfo info{};
            info.context = input.context;
            info.size = input.size;
            info.usage = input.usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            info.memoryProperty = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            return createBuffer(info);
        }

        [[nodiscard]] ErrorDataOutput<void> createGPUTypeBuffer(const CreateInfo &input) noexcept{

            BufferBaseCreateInfo info{};
            info.context = input.context;
            info.size = input.size;
            info.usage = input.usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   

            if constexpr(readable){
                info.memoryProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            }else{
                info.memoryProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            }
            return createBuffer(info);
        }
    };
    
    template<BufferType type , uint8_t readable>
    using pGeneralBuffer2 = std::shared_ptr<GeneralBuffer2<type, readable>>;

    template<typename T>
    requires GPUDataType<T>
    std::span<T> translateToSpan(std::vector<T>& data) noexcept{
        return std::span<T>(data);
    }
}
