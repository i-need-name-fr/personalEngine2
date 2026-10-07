#pragma once

#include "Base_buffer.hpp"
#include <span>
#include <concepts>



namespace mox{

    class HostBuffer;
    using pHostBuffer = std::shared_ptr<HostBuffer>;

    template<typename T>
    concept MassiveTypeData = requires(T t){
        {t.size()};
        {t.data()};
    };

    template<typename T>
    concept GPUDataType = !std::is_pointer_v<T> && std::is_standard_layout_v<T> && !MassiveTypeData<T>;

    class HostBuffer final : public  BufferBase{
    public:

        struct CreateInfo{
            VulkanContext* context;
            VkDeviceSize size =0;
            VkBufferUsageFlags usage;
        };

        HostBuffer() = delete;
        HostBuffer(HostBuffer& input) = delete;
        HostBuffer& operator=(HostBuffer& input) = delete;

        HostBuffer(HostBuffer&& input ) noexcept : BufferBase(std::move(input)){
        }

        HostBuffer& operator=(HostBuffer&& input) noexcept{
            performCopy(std::move(input));
            return *this;
        }

        ~HostBuffer(){}

        explicit HostBuffer(const CreateInfo &info) : BufferBase(info.context){

            BufferBaseCreateInfo data{};
            data.context = info.context;
            data.memoryProperty = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            data.size = info.size;
            data.usage = info.usage;

            const auto result = createBuffer(data);
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }

            vkMapMemory(device , memory , 0 , data.size , 0 , &this->data);
        }

        template<typename T>
        requires GPUDataType<T>
        explicit HostBuffer(const CreateInfo &info , std::vector<T> input):  HostBuffer(info){
            if(!input.empty()){
                const uint32_t size = std::min(info.size , input.size() * sizeof(T));
                memcpy(this->data , input.data() , size);
            }
        }

        explicit HostBuffer(const CreateInfo &info , void* input):  HostBuffer(info){
            if(input){
                memcpy(this->data , input , info.size);
            }
        }

        void recordData(VkDeviceSize inputSize , void*inputData) noexcept{
            if(inputData){
                VkDeviceSize finalSize = std::min(inputSize , bufferSize);
                memcpy(this->data , inputData , finalSize);
            }
        }

        template<typename T>
        requires GPUDataType<T>
        void recordData(std::vector<T> &input){
            if(!input.empty()){
                const VkDeviceSize finalSize = std::min( input.size() * sizeof(T) , bufferSize);
                memcpy(this->data , input.data() , finalSize);
            }
        }

        template<typename T>
        requires GPUDataType<T>
        void recordData(T &input){
            const VkDeviceSize finalSize = std::min( sizeof(T) , bufferSize);
            memcpy(this->data , &input , finalSize);
        }


        template<typename T>
        requires GPUDataType<T>
        T getSingularData() noexcept {
            T var;
            memcpy(&var , this->data , sizeof(T));
            return var;
        }

        template<typename T>
        requires GPUDataType<T>
        std::vector<T> getBufferData(const VkDeviceSize amount) noexcept{
            VkDeviceSize finalSize = std::min(amount * sizeof(T) , bufferSize);
            std::vector<T> buf(amount);
            memcpy(buf.data() , this->data , finalSize);
            return buf;
        }

    };

    using pHostBuffer = std::shared_ptr<HostBuffer>;
}