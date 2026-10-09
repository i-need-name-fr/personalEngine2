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


    template<BufferType type , uint8_t readable>
    class GeneralBuffer2 final : public BufferBase{
    public:

        GeneralBuffer2() = delete;
        GeneralBuffer2(const GeneralBuffer2& input) = delete;
        GeneralBuffer2(GeneralBuffer2& input) = delete;
        GeneralBuffer2& operator=(GeneralBuffer2& input) = delete;
        GeneralBuffer2& operator=(const GeneralBuffer2& input) = delete;

        GeneralBuffer2(GeneralBuffer2&& input) noexcept : BufferBase(std::move(input)){currentAccess = input.currentAccess; currentStage= input.currentStage;}

        GeneralBuffer2& operator=(GeneralBuffer2&& input) noexcept{
            if(this == &input) return *this;
            // performCopy only takes the handles of the other buffer , the buffer and the memory of this one would stay alive for ever
            cleanBuffer();
            performCopy(std::move(input));
            currentAccess = input.currentAccess; currentStage= input.currentStage;
            return *this;
        }


        struct CreateInfo{
            VulkanContext* context;

            VkDeviceSize size;
            VkBufferUsageFlags usage;
        };

        explicit GeneralBuffer2(const CreateInfo &info) : BufferBase(info.context){
            if(info.size == 0) throw std::runtime_error("failed to create generalBuffer , since the size is 0");
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
                        const auto result = copyBufferToBuffer(staging.buffer , this->buffer , requiredSize );
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
                cleanBuffer();
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

                    pBufferBase staging = std::make_shared<BufferBase>(data);

                    
                    auto result = vkMapMemory(info.context->logicalDevice->device , staging->memory , 0 , info.size , 0 , &staging->data);
                    if(result != VK_SUCCESS){
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create general buffer , since the vk map memory failed for staging buffer , error ID -> {}" , static_cast<int32_t>(result))};
                        engineLogger(error);
                        THROW_MESSAGE;
                    }

                    std::memcpy(staging->data , inputData , info.size );
                    

                    const auto result2 = copyBufferToBuffer(staging->buffer , buffer , info.size );
                    if(!result2.has_value()){
                        engineLogger(result2.error());
                        THROW_MESSAGE;
                    }
                }
            }
            catch(std::exception &e){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create general buffer , check the logs please"};
                engineLogger(error);
                cleanBuffer();
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
        
        void fillBuffer(VkCommandBuffer &cmd) noexcept requires(BufferType::Device == type && !readable){
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

        // add new functions
        // update within cmd for readable
        // translate buffer

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

        // writes the data inside of the command buffer ( the barriers are part of it ) .
        // vkCmdUpdateBuffer takes at most 65536 bytes at a time and its size and offset have to be multiples of 4 , so the data is cut into
        // chunks of 65536 bytes , every chunk is written at its own offset . what does not fit in the buffer is not written ,
        // and the bytes after the last multiple of 4 are not written either ( both are logged )
        template<typename T>
        requires GPUDataType<T>
        void updateBufferCmd(VkCommandBuffer& cmd , std::span<T> input , VkAccessFlags2 access , VkPipelineStageFlags2 stage)noexcept requires(readable == true){

            if(!cmd) return;
            if(input.empty()) return;

            constexpr VkDeviceSize chunkSize = 65536u;

            const VkDeviceSize wanted = static_cast<VkDeviceSize>(input.size_bytes());
            // the size of the whole write : not more than the buffer , and a multiple of 4
            const VkDeviceSize total = std::min(wanted , bufferSize) & ~static_cast<VkDeviceSize>(3u);

            if(total != wanted){
                static uint32_t totalMessagesCount = 0;
                if(totalMessagesCount < totalMessageCount){
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("updateBufferCmd writes {} bytes of {} : the buffer has {} bytes and the size has to be a multiple of 4 , message Count {} " , total , wanted , bufferSize , static_cast<uint32_t>(totalMessagesCount))};
                    engineLogger(error);
                    totalMessagesCount++;
                }
            }
            if(total == 0) return;

            // the chunks : the start of every one of them ( the size of every chunk is chunkSize , the last one is what is left )
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(input.data());
            std::vector<const uint8_t*> chunks{};
            chunks.reserve(static_cast<size_t>((total + chunkSize - 1) / chunkSize));
            for(VkDeviceSize offset = 0 ; offset < total ; offset += chunkSize){
                chunks.push_back(bytes + offset);
            }

            auto barrier = translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 1;
            info.imageMemoryBarrierCount = 0;
            info.pMemoryBarriers = nullptr;
            info.pBufferMemoryBarriers = &barrier;
            info.pImageMemoryBarriers = nullptr;
            info.memoryBarrierCount = 0;
            info.pNext = nullptr;
            info.dependencyFlags = 0;

            vkCmdPipelineBarrier2(cmd , &info);

            for(size_t i = 0 ; i < chunks.size() ; i++){
                const VkDeviceSize offset = static_cast<VkDeviceSize>(i) * chunkSize;
                const VkDeviceSize size = std::min(chunkSize , total - offset);
                vkCmdUpdateBuffer(cmd , buffer , offset , size , chunks[i]);
            }

            barrier = translateBuffer(access , stage);
            vkCmdPipelineBarrier2(cmd , &info);
        }

        ~GeneralBuffer2(){}

    private:

        VkAccessFlags2 currentAccess{0};
        VkPipelineStageFlags2 currentStage{VK_PIPELINE_STAGE_2_NONE};

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
