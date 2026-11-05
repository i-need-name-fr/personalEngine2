#pragma once

#include <array>
#include <cmath>
#include <format>
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

            inline static constexpr uint32_t INPUT_PADDED = 16;
            inline static constexpr uint32_t OUTPUT_PADDED = 16;

            inline static constexpr uint32_t HIDDEN_WEIGHT_COUNT = HIDDEN_COUNT * INPUT_PADDED;
            inline static constexpr uint32_t HIDDEN_PARAM_COUNT = HIDDEN_WEIGHT_COUNT + HIDDEN_COUNT;
            inline static constexpr uint32_t OUTPUT_WEIGHT_COUNT = OUTPUT_PADDED * HIDDEN_COUNT;
            inline static constexpr uint32_t OUTPUT_PARAM_COUNT = OUTPUT_WEIGHT_COUNT + OUTPUT_PADDED;
            static_assert(INPUT_COUNT <= INPUT_PADDED && OUTPUT_COUNT <= OUTPUT_PADDED , "the real sizes must fit into the padded ones");

            inline static constexpr uint64_t W1_OFFSET = 0;
            inline static constexpr uint64_t B1_OFFSET = alignUp64(W1_OFFSET + HIDDEN_COUNT * INPUT_PADDED * 2ull);
            inline static constexpr uint64_t HIDDEN_BYTES = alignUp64(B1_OFFSET + HIDDEN_COUNT * 2ull);
            inline static constexpr uint64_t W2_OFFSET = 0;
            inline static constexpr uint64_t B2_OFFSET = alignUp64(W2_OFFSET + OUTPUT_PADDED * HIDDEN_COUNT * 2ull);
            inline static constexpr uint64_t OUTPUT_BYTES = alignUp64(B2_OFFSET + OUTPUT_PADDED * 2ull);
            inline static constexpr uint64_t TRAINING_MATRIX_BYTES = 3584ull;
            inline static constexpr uint64_t B1_GRAD_OFFSET = TRAINING_MATRIX_BYTES;
            inline static constexpr uint64_t B2_GRAD_OFFSET = TRAINING_MATRIX_BYTES;
            inline static constexpr uint64_t W1_TRAIN_OFFSET = HIDDEN_BYTES;
            inline static constexpr uint64_t W2_TRAIN_OFFSET = OUTPUT_BYTES;
            inline static constexpr uint64_t HIDDEN_TOTAL_BYTES = alignUp64(W1_TRAIN_OFFSET + TRAINING_MATRIX_BYTES);
            inline static constexpr uint64_t OUTPUT_TOTAL_BYTES = alignUp64(W2_TRAIN_OFFSET + TRAINING_MATRIX_BYTES);
            static_assert(HIDDEN_BYTES == 2176 && OUTPUT_BYTES == 2112 , "the offsets are copied to coopVector.hlsl ( NRC_W1_TRAIN_OFFSET = HIDDEN_BYTES ... ) : change both");

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
            // the plain float32 row-major matrices : the optimal layouts cannot be read / written by a shader , only vkCmdConvertCooperativeVectorMatrixNV goes between them and these ones
            pDeviceStaticBuffer gradStagingHiddenBuffer{};   // 64 x 16 : the gradients of W1 after the conversion from TRAINING_OPTIMAL ( apply reads )
            pDeviceStaticBuffer gradStagingOutputBuffer{};   // 16 x 64 : the gradients of W2
            pDeviceStaticBuffer weightStagingHiddenBuffer{}; // 64 x 16 : the new W1 ( apply writes ) , the source of the conversion into INFERENCING_OPTIMAL and TRAINING_OPTIMAL
            pDeviceStaticBuffer weightStagingOutputBuffer{}; // 16 x 64 : the new W2

            void recordConvertGradients(VkCommandBuffer cmd) const noexcept
            {
                const std::array<ConvertRecord , 2> records = {{
                    {neuronHiddenGradBuffer->pBuffer , TRAINING_MATRIX_BYTES , VK_COMPONENT_TYPE_FLOAT16_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV , 0 ,
                     gradStagingHiddenBuffer->pBuffer , HIDDEN_WEIGHT_COUNT * sizeof(float) , VK_COMPONENT_TYPE_FLOAT32_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV , INPUT_PADDED * sizeof(float) ,
                     HIDDEN_COUNT , INPUT_PADDED},
                    {neuronPerChannelGradBuffer->pBuffer , TRAINING_MATRIX_BYTES , VK_COMPONENT_TYPE_FLOAT16_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV , 0 ,
                     gradStagingOutputBuffer->pBuffer , OUTPUT_WEIGHT_COUNT * sizeof(float) , VK_COMPONENT_TYPE_FLOAT32_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV , HIDDEN_COUNT * sizeof(float) ,
                     OUTPUT_PADDED , HIDDEN_COUNT}
                }};
                recordConvert(cmd , records.data() , static_cast<uint32_t>(records.size()));
            }

            void recordConvertWeights(VkCommandBuffer cmd) const noexcept
            {
                const std::array<ConvertRecord , 4> records = {{
                    {weightStagingHiddenBuffer->pBuffer , HIDDEN_WEIGHT_COUNT * sizeof(float) , VK_COMPONENT_TYPE_FLOAT32_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV , INPUT_PADDED * sizeof(float) ,
                     neuronHiddenBuffer->pBuffer + W1_OFFSET , HIDDEN_WEIGHT_COUNT * sizeof(uint16_t) , VK_COMPONENT_TYPE_FLOAT16_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV , 0 ,
                     HIDDEN_COUNT , INPUT_PADDED},
                    {weightStagingHiddenBuffer->pBuffer , HIDDEN_WEIGHT_COUNT * sizeof(float) , VK_COMPONENT_TYPE_FLOAT32_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV , INPUT_PADDED * sizeof(float) ,
                     neuronHiddenBuffer->pBuffer + W1_TRAIN_OFFSET , TRAINING_MATRIX_BYTES , VK_COMPONENT_TYPE_FLOAT16_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV , 0 ,
                     HIDDEN_COUNT , INPUT_PADDED},
                    {weightStagingOutputBuffer->pBuffer , OUTPUT_WEIGHT_COUNT * sizeof(float) , VK_COMPONENT_TYPE_FLOAT32_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV , HIDDEN_COUNT * sizeof(float) ,
                     neuronPerChannelBuffer->pBuffer + W2_OFFSET , OUTPUT_WEIGHT_COUNT * sizeof(uint16_t) , VK_COMPONENT_TYPE_FLOAT16_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV , 0 ,
                     OUTPUT_PADDED , HIDDEN_COUNT},
                    {weightStagingOutputBuffer->pBuffer , OUTPUT_WEIGHT_COUNT * sizeof(float) , VK_COMPONENT_TYPE_FLOAT32_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV , HIDDEN_COUNT * sizeof(float) ,
                     neuronPerChannelBuffer->pBuffer + W2_TRAIN_OFFSET , TRAINING_MATRIX_BYTES , VK_COMPONENT_TYPE_FLOAT16_KHR , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV , 0 ,
                     OUTPUT_PADDED , HIDDEN_COUNT}
                }};
                recordConvert(cmd , records.data() , static_cast<uint32_t>(records.size()));
            }

            // the gradients of the matrices ( the whole buffers : the matrix and the bias gradients ) start the next step from zero . the gradients of the hash grid are zeroed by apply itself ( Parameter.grad )
            void recordZeroGradients(VkCommandBuffer cmd) const noexcept
            {
                vkCmdFillBuffer(cmd , neuronHiddenGradBuffer->buffer , 0 , neuronHiddenGradBuffer->getBufferSize() , 0);
                vkCmdFillBuffer(cmd , neuronPerChannelGradBuffer->buffer , 0 , neuronPerChannelGradBuffer->getBufferSize() , 0);
            }
            // END convert commands

        private:

            struct ConvertRecord
            {
                VkDeviceAddress src;
                size_t srcSize;
                VkComponentTypeKHR srcType;
                VkCooperativeVectorMatrixLayoutNV srcLayout;
                size_t srcStride;
                VkDeviceAddress dst;
                size_t dstSize;
                VkComponentTypeKHR dstType;
                VkCooperativeVectorMatrixLayoutNV dstLayout;
                size_t dstStride;
                uint32_t rows;
                uint32_t columns;
            };

            // all the records in ONE command ( the command takes an array ) . the matrices of one call do not depend on each other
            static void recordConvert(VkCommandBuffer cmd , const ConvertRecord* records , const uint32_t count) noexcept
            {
                constexpr uint32_t maxCount = 4;
                std::array<size_t , maxCount> dstSizes{};
                std::array<VkConvertCooperativeVectorMatrixInfoNV , maxCount> infos{};

                for(uint32_t i = 0 ; i < count && i < maxCount ; i++)
                {
                    const ConvertRecord& r = records[i];
                    dstSizes[i] = r.dstSize;

                    VkConvertCooperativeVectorMatrixInfoNV& info = infos[i];
                    info.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
                    info.srcSize = r.srcSize;
                    info.srcData.deviceAddress = r.src;
                    info.pDstSize = &dstSizes[i];
                    info.dstData.deviceAddress = r.dst;
                    info.srcComponentType = r.srcType;
                    info.dstComponentType = r.dstType;
                    info.numRows = r.rows;
                    info.numColumns = r.columns;
                    info.srcLayout = r.srcLayout;
                    info.srcStride = r.srcStride;
                    info.dstLayout = r.dstLayout;
                    info.dstStride = r.dstStride;
                }

                cmdConvertCooperativeVectorMatrixNV(cmd , count < maxCount ? count : maxCount , infos.data());
            }

            
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
                        // hidden layer : the matrix W1 64 x 16 row after row ( the columns over INPUT_COUNT are the padding : zeros , they never get a gradient ) , then the 64 biases
                        _neuronParamsHidden.reserve(HIDDEN_PARAM_COUNT);
                        for(uint32_t n = 0 ; n < HIDDEN_COUNT ; n++)
                        {
                            for(uint32_t c = 0 ; c < INPUT_PADDED ; c++)
                            {
                                ::NRC::Parameter p{};
                                p.grad = 0.0f;
                                p.w = (c < INPUT_COUNT) ? hiddenInit(engine) : 0.0f;
                                p.m = 0.0f;
                                p.v = 0.0f;
                                _neuronParamsHidden.push_back(p);
                            }
                        }
                        for(uint32_t n = 0 ; n < HIDDEN_COUNT ; n++)
                        {
                            ::NRC::Parameter p{};   // the biases start at 0
                            _neuronParamsHidden.push_back(p);
                        }

                        // output layer : the matrix W2 16 x 64 ( the rows over OUTPUT_COUNT are the padding : zeros ) , then the 16 biases
                        _neuronParamsPerChannel.reserve(OUTPUT_PARAM_COUNT);
                        for(uint32_t ch = 0 ; ch < OUTPUT_PADDED ; ch++)
                        {
                            for(uint32_t c = 0 ; c < HIDDEN_COUNT ; c++)
                            {
                                ::NRC::Parameter p{};
                                p.grad = 0.0f;
                                p.w = (ch < OUTPUT_COUNT) ? outputInit(engine) : 0.0f;
                                p.m = 0.0f;
                                p.v = 0.0f;
                                _neuronParamsPerChannel.push_back(p);
                            }
                        }
                        for(uint32_t ch = 0 ; ch < OUTPUT_PADDED ; ch++)
                        {
                            ::NRC::Parameter p{};
                            _neuronParamsPerChannel.push_back(p);
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
                            // the cells of the levels : 8 , 4 , 2 , 1 meters ( T = 8 , 16 , 32 , 64 ) : the grid of every level repeats with the period T * cellSize = 64 meters
                            info.cellSize = 8.0f / static_cast<float>(1u << id);
                            info.T = 8u * (2u << id);
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
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = sizeof(::NRC::Parameter) * _hashGridParamsVector.size();

                        hashGridParamsBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(_hashGridParamsVector));
                    }

                    {
                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = sizeof(float) * _hashGridVector.size();

                        hashGridBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(_hashGridVector));
                    }

                    {

                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = HIDDEN_TOTAL_BYTES;
                        std::vector<uint32_t> zeros(info.size / sizeof(uint32_t) , 0u);

                        neuronHiddenBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(zeros));
                    }


                    {

                        DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = flags;
                        info.size = OUTPUT_TOTAL_BYTES;
                        std::vector<uint32_t> zeros(info.size / sizeof(uint32_t) , 0u);

                        neuronPerChannelBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(zeros));
                    }

                    {
                        {
                            DeviceStaticBuffer::CreateInfo info{};
                            info.context = context;
                            info.usage = flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // TRANSFER_DST : vkCmdFillBuffer zeroes the gradients every step
                            info.size = neuronHiddenBuffer->getBufferSize() * 2;
                            std::vector<uint32_t> zeros(info.size / sizeof(uint32_t) , 0u);
                            neuronHiddenGradBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(zeros));
                        }

                        {
                            DeviceStaticBuffer::CreateInfo info{};
                            info.context = context;
                            info.usage = flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                            info.size = neuronPerChannelBuffer->getBufferSize() * 2;
                            std::vector<uint32_t> zeros(info.size / sizeof(uint32_t) , 0u);
                            neuronPerChannelGradBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(zeros));
                        }

                        {
                            DeviceStaticBuffer::CreateInfo info{};
                            info.context = context;
                            info.usage = flags;

                            std::vector<float> gradsHidden(HIDDEN_WEIGHT_COUNT , 0.0f);
                            info.size = sizeof(float) * gradsHidden.size();
                            gradStagingHiddenBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(gradsHidden));

                            std::vector<float> gradsOutput(OUTPUT_WEIGHT_COUNT , 0.0f);
                            info.size = sizeof(float) * gradsOutput.size();
                            gradStagingOutputBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(gradsOutput));

                            std::vector<float> weightsHidden(HIDDEN_WEIGHT_COUNT);
                            for(uint32_t i = 0 ; i < HIDDEN_WEIGHT_COUNT ; i++) weightsHidden[i] = _neuronParamsHidden[i].w;
                            info.size = sizeof(float) * weightsHidden.size();
                            weightStagingHiddenBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(weightsHidden));

                            std::vector<float> weightsOutput(OUTPUT_WEIGHT_COUNT);
                            for(uint32_t i = 0 ; i < OUTPUT_WEIGHT_COUNT ; i++) weightsOutput[i] = _neuronParamsPerChannel[i].w;
                            info.size = sizeof(float) * weightsOutput.size();
                            weightStagingOutputBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(weightsOutput));
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
                        general.stepCount = 0;
                        std::vector<::NRC::GeneralData> generalVector{general};
                        generalDataBuffer = std::make_shared<DeviceStaticBuffer>(info , translateToSpan(generalVector));
                    }

                    {
                        std::vector<DescriptorLayoutData> data(14);
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
                        // the staging matrices : gradients of W1 , W2 , new weights of W1 , W2
                        data[10] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[11] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[12] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                        data[13] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};


                        DescriptorLayout::CreateInfo info{};
                        info.context = context;
                        info.stage = VK_SHADER_STAGE_ALL;
                        info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                        info.data = data;
                        info.pNext = nullptr;

                        layout = std::make_shared<DescriptorLayout>(info);
                    }

                    {
                        std::vector<DescriptorData2> data(14);
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
                        data[10] = mox::DescriptorData2::createData(gradStagingHiddenBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 10 , 0);
                        data[11] = mox::DescriptorData2::createData(gradStagingOutputBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 11 , 0);
                        data[12] = mox::DescriptorData2::createData(weightStagingHiddenBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 12 , 0);
                        data[13] = mox::DescriptorData2::createData(weightStagingOutputBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 13 , 0);

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
                gradStagingHiddenBuffer = std::move(input.gradStagingHiddenBuffer);
                gradStagingOutputBuffer = std::move(input.gradStagingOutputBuffer);
                weightStagingHiddenBuffer = std::move(input.weightStagingHiddenBuffer);
                weightStagingOutputBuffer = std::move(input.weightStagingOutputBuffer);
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
                gradStagingHiddenBuffer = nullptr;
                gradStagingOutputBuffer = nullptr;
                weightStagingHiddenBuffer = nullptr;
                weightStagingOutputBuffer = nullptr;

                layout = nullptr;
            }
        };

        using pNRCManager = std::shared_ptr<NRCManager>;


    }
}