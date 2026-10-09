#pragma once

#include "AccelerationStructureBase.hpp"

namespace mox{

    struct BLAS2_layout{

        uint32_t indexOffset;
        uint32_t indexCount;
        uint32_t vertexCount;
        uint32_t firstVertex;
    };

    class BLAS2 final : public AccelerationStructure{
    public:

        struct CreateInfo{

            VulkanContext* context;

            BufferBase* vertexBuffer;
            BufferBase* indicesBuffer;

            BLAS2_layout layout;

        };

        BLAS2() = delete;
        BLAS2(BLAS2& input) = delete;
        BLAS2(const BLAS2& input) = delete;
        BLAS2(BLAS2&& input) noexcept : AccelerationStructure(std::move(input)){}

        BLAS2& operator=(BLAS2& input) = delete;
        BLAS2& operator=(const BLAS2& input) = delete;
        BLAS2& operator=(BLAS2&& input) noexcept {
            processCopy(std::move(input));
            return *this;
        }

        BLAS2(const CreateInfo &info) : AccelerationStructure(info.context){
            const auto result = createBLAS(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                processDelete();
                THROW_MESSAGE;
            }
        }

        ~BLAS2(){}

    private:

        [[nodiscard]] ErrorDataOutput<void> createBLAS(const CreateInfo &info) noexcept {

            try{

                if(!info.vertexBuffer || !info.indicesBuffer){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create BLAS , since the vertex or the index buffer is nullptr"});
                }

                if(!info.vertexBuffer->pBuffer || !info.indicesBuffer->pBuffer){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create BLAS , since the vertex or the index buffer has no device address"});
                }

                if(info.layout.indexCount < 3 || info.layout.indexCount % 3 != 0){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create BLAS , the index count has to be a multiple of 3 , got -> {}" , info.layout.indexCount)});
                }

                if(info.layout.vertexCount == 0){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create BLAS , since the vertex count is 0"});
                }

                const uint32_t primitiveCount = info.layout.indexCount / 3;

                VkAccelerationStructureGeometryKHR geometry{};
                geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
                geometry.pNext = nullptr;
                geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
                geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                geometry.geometry.triangles.indexData.deviceAddress = info.indicesBuffer->pBuffer;
                geometry.geometry.triangles.maxVertex = info.layout.vertexCount - 1;
                geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                geometry.geometry.triangles.vertexStride = sizeof(Vertex);
                geometry.geometry.triangles.vertexData.deviceAddress =info.vertexBuffer->pBuffer;
                geometry.geometry.triangles.pNext = nullptr;

                VkAccelerationStructureBuildGeometryInfoKHR build{};
                build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                build.flags = 0;
                build.geometryCount = 1;
                build.dstAccelerationStructure = 0;
                build.srcAccelerationStructure = 0;
                build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                build.pGeometries = &geometry;
                build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

                VkAccelerationStructureBuildSizesInfoKHR sizes{};
                sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

                getAccelerationStructureBuildSizesKHR(device , VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR , &build ,&primitiveCount , &sizes  );

                {
                    // named differently from the outer `info` - the inner one used to shadow it , so
                    // `info.context = info.context` left the context nullptr and the buffer never got created
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo bufferInfo{};
                    bufferInfo.context = info.context;
                    bufferInfo.usage= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    bufferInfo.size = sizes.accelerationStructureSize;

                    this->build = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(bufferInfo);
                }

                {
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo bufferInfo{};
                    bufferInfo.context = info.context;
                    bufferInfo.usage= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    // the FIRST build needs buildScratchSize - min() with updateScratchSize
                    // ( which is usually smaller ) left the scratch buffer too small for it
                    bufferInfo.size = std::max(sizes.buildScratchSize , sizes.updateScratchSize);
                    this->scratch = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(bufferInfo);
                }

                build.scratchData.deviceAddress = scratch->pBuffer;

                {
                    VkAccelerationStructureCreateInfoKHR info{};
                    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                    info.createFlags = 0;
                    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                    info.buffer = this->build->buffer;
                    info.size = sizes.accelerationStructureSize;
                    info.pNext = nullptr;
                    info.offset = 0;

                    auto result = createAccelerationStructureKHR(device , &info , nullptr , &AS);
                    if(result != VK_SUCCESS){
                        ErrorDataType const error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create BLAS as acceleration structure , error ID - > {}" , static_cast<int32_t>(result))};
                        return std::unexpected(error);
                    }

                    VkAccelerationStructureDeviceAddressInfoKHR add{};
                    add.accelerationStructure = AS;
                    add.pNext = nullptr;
                    add.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;

                    pAS = getAccelerationStructureDeviceAddressKHR(device , &add);
                }


                build.dstAccelerationStructure= AS;

                VkAccelerationStructureBuildRangeInfoKHR range{};

                // firstVertex is added to every index value , primitiveOffset is a BYTE offset into the
                // index buffer ( indexOffset is counted in indices )
                range.firstVertex = info.layout.firstVertex;
                range.primitiveCount = primitiveCount;
                range.primitiveOffset = info.layout.indexOffset * sizeof(uint32_t);
                range.transformOffset = 0;

                const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;

                info.context->queueManager->Begin();

                cmdFunctionType function = [&](VkCommandBuffer &cmd) -> void{
                    VkMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    barrier.pNext = nullptr;
                    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

                    vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);

                    mox::cmdBuildAccelerationStructuresKHR(cmd , 1 , &build , &ranges);

                    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

                    vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);
                };

                info.context->queueManager->recordCmd(function , VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

                {
                    const auto result = info.context->queueManager->End();
                    if(!result.has_value()) return std::unexpected(result.error());
                }

                this->scratch = nullptr;
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create BLAS, lol"});
            }

            return {};
        }
    };


    using pBLAS2 = std::shared_ptr<BLAS2>;
}