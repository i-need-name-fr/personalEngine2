#pragma once

#include "AccelerationStructureBase.hpp"
#include "../Object2/Mesh2.hpp"

namespace mox{

    class TLAS2 final : public AccelerationStructure{
    public:

        TLAS2() = delete;
        TLAS2(TLAS2& input) = delete;
        TLAS2(const TLAS2& input) = delete;
        TLAS2& operator=(TLAS2& input) = delete;
        TLAS2& operator=(const TLAS2& input) = delete;

        TLAS2(TLAS2&& input) noexcept : AccelerationStructure(std::move(input)){
            if(TLAS_matrix != input.TLAS_matrix)TLAS_matrix = std::move(input.TLAS_matrix);
            builtInstanceCount = input.builtInstanceCount;
        }
        TLAS2& operator=(TLAS2&& input) noexcept{
            if(input.AS == this->AS) return *this;
            processCopy(std::move(input));
            TLAS_matrix = std::move(input.TLAS_matrix);
            builtInstanceCount = input.builtInstanceCount;
            return *this;
        }

        explicit TLAS2(VulkanContext* context) : AccelerationStructure(context){
            {
                const auto result = createTLAS(context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    processDelete();
                    TLAS_matrix = nullptr;
                    THROW_MESSAGE;
                }
            }
        }

        ~TLAS2(){
            TLAS_matrix = nullptr;
        }

        void updateTLAS(VkCommandBuffer& cmd) noexcept{
            if(!cmd) return;

            // the number of the instances that were written ( the deleted ones are not in the buffer ) . it used to be the number of
            // layouts , deleted ones included : the TLAS read the old record that was left behind the written ones
            const uint32_t primitiveCount = uploadGeometry();

            VkAccelerationStructureGeometryKHR geometry{};
            geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometry.flags = 0;
            geometry.pNext = nullptr;
            geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            geometry.geometry.instances.data.deviceAddress = TLAS_matrix->pBuffer;
            geometry.geometry.instances.arrayOfPointers = VK_FALSE;
            geometry.geometry.instances.pNext = nullptr;

            VkAccelerationStructureBuildGeometryInfoKHR build{};
            build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            build.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            build.geometryCount = 1;
            // an UPDATE has to have the same number of the instances as the build that it updates , with another number the TLAS is built again
            // ( the scratch buffer and the storage were made for Mesh2::maxInstanceCount , so a build always fits )
            const bool canUpdate = (primitiveCount == builtInstanceCount);
            build.mode = canUpdate ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            build.pGeometries = &geometry;
            build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            build.scratchData.deviceAddress = scratch->pBuffer;

            build.dstAccelerationStructure= AS;
            build.srcAccelerationStructure = canUpdate ? AS : VK_NULL_HANDLE;
            builtInstanceCount = primitiveCount;

            VkAccelerationStructureBuildRangeInfoKHR range{};

            range.firstVertex = 0;
            range.primitiveCount = primitiveCount;
            range.primitiveOffset = 0;
            range.transformOffset = 0;

            const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;

            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.pNext = nullptr;
            barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

            vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);

            mox::cmdBuildAccelerationStructuresKHR(cmd , 1 , &build , &ranges);

            barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

            vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);
        }

    private:

        pGeneralBuffer2<BufferType::Device , true> TLAS_matrix{nullptr};

        // how many instances the TLAS was built with ( an UPDATE needs the same number )
        uint32_t builtInstanceCount{0};

        // returns the number of the instances that were written into the buffer
        uint32_t uploadGeometry() noexcept{


            std::vector<VkAccelerationStructureInstanceKHR> instances{};
            for(uint32_t i = 0 ; i <   Mesh2::totalInstanceLayouts.size() ; i++){
                if(!Mesh2::totalInstanceLayouts[i].needsToBeDraw) continue;
                const uint32_t instanceID = Mesh2::totalMeshesData[Mesh2::totalInstanceLayouts[i].meshDataIndex].instanceIndex;
                if(Mesh2::BLAS_storage.size() <= Mesh2::totalInstanceLayouts[i].BLAS_index){
                    std::cerr << "failed to upload geometry for TLAS , since one of the instance , ID " << Mesh2::totalInstanceLayouts[i].BLAS_index << " is out of range\n";
                    continue;
                }

                if(!Mesh2::BLAS_storage[Mesh2::totalInstanceLayouts[i].BLAS_index]){
                    std::cerr << "failed to upload geometry for TLAS , since one of the instance , ID " << Mesh2::totalInstanceLayouts[i].BLAS_index << " contains null BLAS2 \n";
                    continue;
                }
                VkAccelerationStructureInstanceKHR data{};
                data.accelerationStructureReference = Mesh2::BLAS_storage[Mesh2::totalInstanceLayouts[i].BLAS_index]->pAS;
                data.instanceCustomIndex = i;
                data.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                data.instanceShaderBindingTableRecordOffset = 0;
                data.mask = 0xFF;
                data.transform = matrixTransform34(Mesh2::instances[instanceID]);

                instances.push_back(data);
            }

            TLAS_matrix->recordData(translateToSpan(instances));
            return static_cast<uint32_t>(instances.size());
        }

        [[nodiscard]] ErrorDataOutput<void> createTLAS(VulkanContext* context) noexcept{

            try{

                {
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(VkAccelerationStructureInstanceKHR) * Mesh2::maxInstanceCount;
                    info.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    
                    TLAS_matrix = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }

                const uint32_t primitiveCount = uploadGeometry();
                builtInstanceCount = primitiveCount;

                VkAccelerationStructureGeometryKHR geometry{};
                geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                geometry.flags = 0;
                geometry.pNext = nullptr;
                geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
                geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
                geometry.geometry.instances.data.deviceAddress = TLAS_matrix->pBuffer;
                geometry.geometry.instances.arrayOfPointers = VK_FALSE;
                geometry.geometry.instances.pNext = nullptr;
                

                VkAccelerationStructureBuildGeometryInfoKHR build{};
                build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                build.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                build.geometryCount = 1;
                build.dstAccelerationStructure = 0;
                build.srcAccelerationStructure = 0;
                build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                build.pGeometries = &geometry;
                build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

                VkAccelerationStructureBuildSizesInfoKHR sizes{};
                sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

                getAccelerationStructureBuildSizesKHR(device , VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR , &build ,&Mesh2::maxInstanceCount , &sizes  );

                {
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.usage= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.size = sizes.accelerationStructureSize;

                    this->build = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info);
                }

                {
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.usage= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.size = std::max(sizes.buildScratchSize , sizes.updateScratchSize);

                    this->scratch = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info);
                }

                build.scratchData.deviceAddress = scratch->pBuffer;

                {
                    VkAccelerationStructureCreateInfoKHR info{};
                    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                    info.createFlags = 0;
                    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
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

                range.firstVertex = 0;
                range.primitiveCount = primitiveCount;
                range.primitiveOffset = 0;
                range.transformOffset = 0;

                const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;

                context->queueManager->Begin();

                cmdFunctionType function = [&](VkCommandBuffer &cmd) -> void{
                    VkMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    barrier.pNext = nullptr;
                    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

                    vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);

                    mox::cmdBuildAccelerationStructuresKHR(cmd , 1 , &build , &ranges);

                    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

                    vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);
                };

                context->queueManager->recordCmd(function , VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);


                {
                    const auto result = context->queueManager->End();
                    if(!result.has_value()) return std::unexpected(result.error());
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create BLAS, lol"});
            }

            return {};
        }

        
    };

    using pTLAS2 = std::shared_ptr<TLAS2>;
}