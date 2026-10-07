#pragma once

#include "Base_buffer.hpp"
#include "CPU_buffer.hpp"

namespace mox{

    class DeviceBuffer;
    using pDeviceBuffer = std::shared_ptr<DeviceBuffer>;

    class DeviceBuffer final : public BufferBase{
    public:

        

        DeviceBuffer() = delete;

        DeviceBuffer(DeviceBuffer& input) = delete;
        DeviceBuffer& operator=(DeviceBuffer& input) = delete;

        DeviceBuffer(DeviceBuffer&& input) noexcept : BufferBase(std::move(input)){}

        DeviceBuffer& operator=(DeviceBuffer&& input) noexcept {
            performCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context;
            VkDeviceSize size =0;
            VkBufferUsageFlags usage;
        };

        explicit DeviceBuffer(const CreateInfo &info) : BufferBase(info.context){

            BufferBaseCreateInfo data{};
            data.context = info.context;
            data.memoryProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            data.size = info.size;
            data.usage = info.usage;

            auto result = createBuffer(data);
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        template<typename T>
        requires GPUDataType<T>
        explicit DeviceBuffer(const CreateInfo &info , std::vector<T> &inputData) : BufferBase(info.context){
            // now we need a staging
            mox::HostBuffer::CreateInfo data{};
            data.context = info.context;
            data.size = info.size;
            data.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

            mox::HostBuffer staging(data , inputData);

            {
                BufferBaseCreateInfo data{};
                data.context = info.context;
                data.memoryProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                data.size = info.size;
                data.usage = info.usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

                auto result = createBuffer(data);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }

            const auto result = copyBufferToBuffer(staging.buffer , this->buffer , info.size);
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        explicit DeviceBuffer(const CreateInfo &info , void* inputData) : BufferBase(info.context){
            // now we need a staging
            mox::HostBuffer::CreateInfo data{};
            data.context = info.context;
            data.size = info.size;
            data.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

            mox::HostBuffer staging(data , inputData);

            {
                BufferBaseCreateInfo data{};
                data.context = info.context;
                data.memoryProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                data.size = info.size;
                data.usage = info.usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

                auto result = createBuffer(data);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }

            const auto result = copyBufferToBuffer(staging.buffer , this->buffer , info.size);
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }
    
    };

    using pDeviceBuffer = std::shared_ptr<DeviceBuffer>;

}