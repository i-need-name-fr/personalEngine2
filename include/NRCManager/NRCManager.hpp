#pragma once

#include "../vulkanContext/vulkanBase.hpp"
#include "../Buffer2/GeneralBuffer.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"

namespace mox
{
    namespace NRC
    {

        constexpr uint64_t alignUp64(const uint64_t value) noexcept
        {
            return (value + 63ull) & ~63ull;
        }

        class NRCManager final
        {
        public:

            SetTypicalOperators(NRCManager)
            SetTypicalTransferOperators(NRCManager)

            struct CreateInfo
            {
                VulkanContext* context;
            };

            explicit NRCManager(const CreateInfo &info)
            {
                {
                    const auto result = checkRequirements(VkReqTypeQueueManager | VkReqTypeLogicalDevice | VkReqTypePhysicalDevice , info.context);
                    if(!result.has_value()){
                        engineLogger(result);
                        THROW_MESSAGE;
                    }
                }

                {
                    const auto result = _implProcessInit(info);
                    if(!result.has_value()){
                        engineLogger(result);
                        processDelete();
                        THROW_MESSAGE;
                    }
                }
            }

            ~NRCManager() noexcept
            {
                processDelete();
            }

            inline static constexpr uint32_t GRID_LEVELS = 4;
            inline static constexpr uint32_t F = 2;

            inline static constexpr uint32_t INPUT_COUNT = GRID_LEVELS * F + 3;   // 8 + 3 = 11 ( roughness / metallic come with the specular )
            inline static constexpr uint32_t HIDDEN_COUNT = 64;
            inline static constexpr uint32_t OUTPUT_COUNT = 3;
            inline static constexpr uint32_t PARAMS_PER_HIDDEN_NEURON = INPUT_COUNT + 1;
            inline static constexpr uint32_t PARAMS_PER_CHANNEL = HIDDEN_COUNT + 1;

            inline static constexpr uint32_t INPUT_PADDED = 16;
            inline static constexpr uint32_t OUTPUT_PADDED = 16;

            inline static constexpr uint64_t W1_OFFSET = 0;
            inline static constexpr uint64_t B1_OFFSET = alignUp64(W1_OFFSET + HIDDEN_COUNT * INPUT_PADDED * 2ull);
            inline static constexpr uint64_t HIDDEN_BYTES = alignUp64(B1_OFFSET + HIDDEN_COUNT * 2ull);
            inline static constexpr uint64_t W2_OFFSET = 0;
            inline static constexpr uint64_t B2_OFFSET = alignUp64(W2_OFFSET + OUTPUT_PADDED * HIDDEN_COUNT * 2ull);
            inline static constexpr uint64_t OUTPUT_BYTES = alignUp64(B2_OFFSET + OUTPUT_PADDED * 2ull);
            // the matrices in the OPTIMAL layout may take more than rows * columns * 2 bytes : until the real size is asked from vkConvertCooperativeVectorMatrixNV
            // the buffers have twice the room
            inline static constexpr uint64_t OPTIMAL_LAYOUT_MARGIN = 2;

            pDescriptorLayout layout{};
            pDescriptorBuffer2 dbuffer{};

            pDeviceStaticBuffer hashGridBuffer{}; // contains all the final result of hash grid
            pDeviceStaticBuffer hashGridParamsBuffer{}; // contains all background "wheels"
            pDeviceStaticBuffer hashGridGeneralBuffer{}; // contains general info about each hash grid
            pDeviceStaticBuffer neuronParamsHiddenBuffer{};
            pDeviceStaticBuffer neuronHiddenBuffer{}; // required to be read by the driver
            pDeviceStaticBuffer neuronParamsPerChannelBuffer{};
            pDeviceStaticBuffer neuronPerChannelBuffer{}; // also required to be read by the driver , but for RGB channels
            pDeviceStaticBuffer neuronHiddenGradBuffer{}; // the gradients of neuronHiddenBuffer ( float32 , the shader ADDS into it , zeros at the start )
            pDeviceStaticBuffer neuronPerChannelGradBuffer{}; // the same for neuronPerChannelBuffer
            pDeviceStaticBuffer pointersBuffer{}; // the ADDRESSES of the four buffers above ( NRC::Pointers ) : the cooperative vector reads the matrices by the address
            pDeviceStaticBuffer trainingSampleBuffer{};
            pDeviceStaticBuffer generalDataBuffer{}; // one NRC::GeneralData : width , height of the training grid ( half resolution ) and validCount ( zero it before the trace pass )

        private:
            
            [[nodiscard]] ErrorDataOutput<void> _implProcessInit(const CreateInfo &input) noexcept
            {

                VulkanContext* context = input.context;

                try
                {   
                    const VkBufferUsageFlags flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                    std::vector<float> _hashGridVector{};
                    std::vector<::NRC::Parameter> _hashGridParamsVector{};
                    std::vector<::NRC::Parameter> _neuronParamsHidden{};
                    std::vector<::NRC::Parameter> _neuronParamsPerChannel{};
                    std::vector<::NRC::HashGridInfo> _gridInfo{};

                    static std::random_device dv{};
                    static std::mt19937 engine(dv());
                    std::uniform_real_distribution<float> random(0.00001f, 0.0001f);
                    const float hiddenLimit = std::sqrt(6.0f / static_cast<float>(INPUT_COUNT));
                    const float outputLimit = std::sqrt(6.0f / static_cast<float>(HIDDEN_COUNT));
                    std::uniform_real_distribution<float> hiddenInit(-hiddenLimit, hiddenLimit);
                    std::uniform_real_distribution<float> outputInit(-outputLimit, outputLimit);
                    
                    {
                        const VkExtent2D extent = {context->sc->extent.width / 2 , context->sc->extent.height / 2};
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        const size_t sampleCount = static_cast<size_t>(extent.width) * extent.height * MOX_NRC_MAX_BOUNCE_COUNT;
                        info.size = sizeof(::NRC::_TrainingSample) * sampleCount;

                        ::NRC::_TrainingSample samp{};
                        samp.normal = glm::vec3(0.0f);
                        samp.position = glm::vec3(0.0f);
                        samp.valid = MOX_NRC_NON_VALID_SAMPLE;
                        samp.radiance = glm::vec3(0.0f);

                        std::vector<::NRC::_TrainingSample> samps(sampleCount , samp);

                        trainingSampleBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(samps));
                    }



                    {
                        for(uint32_t n = 0 ; n < HIDDEN_COUNT ; n++)
                        {
                            for(uint32_t c = 0 ; c < PARAMS_PER_HIDDEN_NEURON ; c++)
                            {
                                ::NRC::Parameter p{};
                                p.grad = 0.0f;
                                // the bias is the last one of the neuron and starts at 0
                                p.w = (c == INPUT_COUNT) ? 0.0f : hiddenInit(engine);
                                p.m = 0.0f;
                                p.v = 0.0f;
                                _neuronParamsHidden.push_back(p);
                            }
                        }

                        // the output layer : per channel ( R , G , B ) [ w0 .. w63 , bias ]
                        for(uint32_t ch = 0 ; ch < OUTPUT_COUNT ; ch++)
                        {
                            for(uint32_t c = 0 ; c < PARAMS_PER_CHANNEL ; c++)
                            {
                                ::NRC::Parameter p{};
                                p.grad = 0.0f;
                                p.w = (c == HIDDEN_COUNT) ? 0.0f : outputInit(engine);
                                p.m = 0.0f;
                                p.v = 0.0f;
                                _neuronParamsPerChannel.push_back(p);
                            }
                        }

                        {
                            DeviceStaticBuffer::CreateInfo info{};
                            info.context = context;
                            info.usage = flags;
                            info.size = sizeof(::NRC::Parameter) * _neuronParamsHidden.size();

                            neuronParamsHiddenBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(_neuronParamsHidden));
                        }

                        {
                            DeviceStaticBuffer::CreateInfo info{};
                            info.context = context;
                            info.usage = flags;
                            info.size = sizeof(::NRC::Parameter) * _neuronParamsPerChannel.size();

                            neuronParamsPerChannelBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(_neuronParamsPerChannel));
                        }
                    }

                    {
                        for(uint32_t id = 0 ; id < GRID_LEVELS ; id++)
                        {
                            ::NRC::HashGridInfo info{};
                            info.cellSize = 16.0f / static_cast<float>(1u << id);
                            info.T = 4u * (2u << id);
                            info.offset = _hashGridParamsVector.size();
                            _gridInfo.push_back(info);

                            for(uint32_t z = 0 ; z < info.T ; z++)
                            for(uint32_t y = 0 ; y < info.T ; y++)
                            for(uint32_t x = 0 ; x < info.T ; x++)
                            {
                                const float x1 = random(engine);
                                const float x2 = random(engine);

                                _hashGridVector.push_back(x1);
                                _hashGridVector.push_back(x2);

                                // Adam starts with no memory : m = v = 0
                                ::NRC::Parameter xp1{} , xp2{};
                                xp1.grad = 0.0f;
                                xp1.v = 0.0f;
                                xp1.w = x1;
                                xp1.m = 0.0f;

                                xp2.grad = 0.0f;
                                xp2.v = 0.0f;
                                xp2.w = x2;
                                xp2.m = 0.0f;
                                _hashGridParamsVector.push_back(xp1);
                                _hashGridParamsVector.push_back(xp2);
                            }
                        }
                    }

                    {
                        // grid info buffer
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = sizeof(::NRC::HashGridInfo) * GRID_LEVELS;

                        hashGridGeneralBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(_gridInfo));
                    }

                    {
                        // grid param buffer
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = sizeof(::NRC::Parameter) * _hashGridParamsVector.size();

                        hashGridParamsBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(_hashGridParamsVector));
                    }

                    {
                        // grid param buffer
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = sizeof(float) * _hashGridVector.size();

                        hashGridBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(_hashGridVector));
                    }

                    {

                        // W1 and b1 in float16 ( 2176 bytes ) , the conversion command fills it from the float32 matrices
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = HIDDEN_BYTES * OPTIMAL_LAYOUT_MARGIN;

                        neuronHiddenBuffer = std::make_shared<DeviceStaticBuffer>(info);
                    }


                    {

                        // W2 ( 16 rows , 3 of them used ) and b2 in float16 ( 2112 bytes )
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = OUTPUT_BYTES * OPTIMAL_LAYOUT_MARGIN;

                        neuronPerChannelBuffer = std::make_shared<DeviceStaticBuffer>(info);
                    }

                    {
                        // the gradients : the same matrices , but float32 ( twice the bytes ) and they MUST start with zeros ( a device buffer is not zeroed by itself )
                        {
                            DeviceStaticBuffer::CreateInfo info{};
                            info.context = context;
                            info.usage = flags;
                            info.size = neuronHiddenBuffer->getBufferSize() * 2;
                            std::vector<uint32_t> zeros(info.size / sizeof(uint32_t) , 0u);
                            neuronHiddenGradBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(zeros));
                        }

                        {
                            DeviceStaticBuffer::CreateInfo info{};
                            info.context = context;
                            info.usage = flags;
                            info.size = neuronPerChannelBuffer->getBufferSize() * 2;
                            std::vector<uint32_t> zeros(info.size / sizeof(uint32_t) , 0u);
                            neuronPerChannelGradBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(zeros));
                        }

                        // the addresses ( every buffer was made with SHADER_DEVICE_ADDRESS , the address is in pBuffer )
                        ::NRC::Pointers pointers{};
                        pointers.hiddenWeights = neuronHiddenBuffer->pBuffer;
                        pointers.outputWeights = neuronPerChannelBuffer->pBuffer;
                        pointers.hiddenGrads = neuronHiddenGradBuffer->pBuffer;
                        pointers.outputGrads = neuronPerChannelGradBuffer->pBuffer;
                        std::vector<::NRC::Pointers> pointersVector{pointers};

                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = sizeof(::NRC::Pointers);
                        pointersBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(pointersVector));
                    }

                    {
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // TRANSFER_DST : vkCmdFillBuffer / vkCmdUpdateBuffer can reset validCount every frame
                        info.size = sizeof(::NRC::GeneralData);
                        ::NRC::GeneralData general{};
                        general.width = context->sc->extent.width / 2;
                        general.height = context->sc->extent.height / 2;
                        general.validCount = 0;
                        general.padding = 0.0f;
                        std::vector<::NRC::GeneralData> generalVector{general};
                        generalDataBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(generalVector));
                    }

                    {
                        std::vector<DescriptorLayoutData> data(10);
                        data[0] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[1] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[2] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[3] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[4] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[5] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[6] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[7] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        // the addresses of the matrices ( NrcPointers in coopVector.hlsl ) : a UNIFORM buffer , the shader reads the four addresses from it
                        data[8] = {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , .count = 1};
                        data[9] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};


                        DescriptorLayout::CreateInfo info{};
                        info.context = context;
                        info.stage = VK_SHADER_STAGE_ALL;
                        info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                        info.data = data;
                        info.pNext = nullptr;

                        layout = std::make_shared<DescriptorLayout>(info);
                    }

                    {
                        std::vector<DescriptorData2> data(10);
                        data[0] = mox::DescriptorData2::createData(trainingSampleBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0);
                        data[1] = mox::DescriptorData2::createData(hashGridBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0);
                        data[2] = mox::DescriptorData2::createData(hashGridParamsBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 2 , 0);
                        data[3] = mox::DescriptorData2::createData(hashGridGeneralBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0);
                        data[4] = mox::DescriptorData2::createData(neuronParamsHiddenBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0);
                        data[5] = mox::DescriptorData2::createData(neuronParamsPerChannelBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 5 , 0);
                        data[6] = mox::DescriptorData2::createData(neuronHiddenBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 6 , 0);
                        data[7] = mox::DescriptorData2::createData(neuronPerChannelBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 7 , 0);
                        data[8] = mox::DescriptorData2::createData(pointersBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 8 , 0);
                        data[9] = mox::DescriptorData2::createData(generalDataBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 9 , 0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = layout.get();

                        dbuffer = std::make_shared<DescriptorBuffer2>(info);
                    }
                }
                catch(std::exception &e)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to impl init for NRC manager"});
                }
                return {};
            }

            void processCopy(NRC::NRCManager&& input) noexcept
            {
                layout = std::move(input.layout);
                dbuffer = std::move(input.dbuffer);
                hashGridBuffer = std::move(input.hashGridBuffer);
                hashGridParamsBuffer = std::move(input.hashGridParamsBuffer);
                hashGridGeneralBuffer = std::move(input.hashGridGeneralBuffer);
                neuronParamsHiddenBuffer = std::move(input.neuronParamsHiddenBuffer);
                neuronHiddenBuffer = std::move(input.neuronHiddenBuffer);
                neuronParamsPerChannelBuffer = std::move(input.neuronParamsPerChannelBuffer);
                neuronPerChannelBuffer = std::move(input.neuronPerChannelBuffer);
                neuronHiddenGradBuffer = std::move(input.neuronHiddenGradBuffer);
                neuronPerChannelGradBuffer = std::move(input.neuronPerChannelGradBuffer);
                pointersBuffer = std::move(input.pointersBuffer);
                trainingSampleBuffer = std::move(input.trainingSampleBuffer);
                generalDataBuffer = std::move(input.generalDataBuffer);
            }

            void processDelete() noexcept
            {
                dbuffer = nullptr;

                hashGridBuffer = nullptr;
                hashGridGeneralBuffer = nullptr;
                hashGridParamsBuffer = nullptr;
                neuronParamsHiddenBuffer = nullptr;
                neuronPerChannelBuffer = nullptr;
                neuronHiddenBuffer = nullptr;
                neuronPerChannelBuffer = nullptr;
                neuronParamsPerChannelBuffer = nullptr;
                neuronHiddenGradBuffer = nullptr;
                neuronPerChannelGradBuffer = nullptr;
                pointersBuffer = nullptr;
                trainingSampleBuffer = nullptr;
                generalDataBuffer = nullptr;

                layout = nullptr;
            }
        };

        using pNRCManager = std::shared_ptr<NRCManager>;


    }
}