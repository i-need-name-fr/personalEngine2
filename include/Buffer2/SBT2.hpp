#pragma once

#include "GeneralBuffer.hpp"

namespace mox{

    // =============================== INFO ===============================
    // SBT2 - the shader binding table of a ray tracing pipeline
    //
    // CHANGES from SBT :
    // -- the table is a GeneralBuffer2 ( the upload through the staging is done by it ) , SBT2 does not inherit from a buffer any more
    // -- the groups are described by their TYPE in the order of the pipeline : a hit group is ONE group that can hold a closest hit and an
    //    any hit shader at once ( the old ClosestHit / AnyHit were two names of the same kind of group , and AnyHit was refused )
    // -- the callable region exists now
    // -- the properties of the device are read once , by a function static ( the old one was an uninitialised static flag )
    // -- the address of the table is checked to be aligned for shaderGroupBaseAlignment
    //
    // the layout of the table : [ ray generation | miss | hit | callable ] , every region starts at a multiple of shaderGroupBaseAlignment
    // =============================== END ===============================

    enum class SBTGroupType{
        RayGeneration,
        Miss,
        // closest hit and any hit shaders live in a hit group ( one group of the pipeline )
        Hit,
        Callable
    };

    class SBT2 final{
    public:

        SBT2() = delete;
        SBT2(SBT2& input) = delete;
        SBT2(const SBT2& input) = delete;
        SBT2& operator=(SBT2& input) = delete;
        SBT2& operator=(const SBT2& input) = delete;

        SBT2(SBT2&& input) noexcept{
            processCopy(std::move(input));
        }

        SBT2& operator=(SBT2&& input) noexcept{
            if(this == &input) return *this;
            processCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context = nullptr;

            VkPipeline pipeline = nullptr;

            // groupTypes[i] is the type of the group i of the pipeline ( exactly one RayGeneration ) , the number of the groups of the
            // pipeline has to be the same . inside one type the order of the groups is kept : that order is what the miss index , the
            // hit group index and the callable index in the shaders refer to
            std::vector<SBTGroupType> groupTypes{};
        };

        explicit SBT2(const CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }

            if(!info.pipeline){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create SBT2 , since the pipeline is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(info.groupTypes.empty()){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create SBT2 , since the group types are empty"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            const auto result = createSBT(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                processDelete();
                THROW_MESSAGE;
            }
        }

        ~SBT2() noexcept{
            processDelete();
        }

        // the regions as vkCmdTraceRaysKHR wants them
        [[nodiscard]] const VkStridedDeviceAddressRegionKHR& getRayGenRegion() const noexcept{ return rayGen; }
        [[nodiscard]] const VkStridedDeviceAddressRegionKHR& getMissRegion() const noexcept{ return miss; }
        [[nodiscard]] const VkStridedDeviceAddressRegionKHR& getHitRegion() const noexcept{ return hit; }
        [[nodiscard]] const VkStridedDeviceAddressRegionKHR& getCallableRegion() const noexcept{ return callable; }

        // the device address of the start of the whole table
        [[nodiscard]] VkDeviceAddress getAddress() const noexcept{
            return table ? table->pBuffer : 0;
        }

        // the buffer of the table is read only after the creation , the graph does not have to translate it
        [[nodiscard]] GeneralBuffer2<BufferType::Device , false>* getBuffer() const noexcept{
            return table.get();
        }

        void traceRays(VkCommandBuffer &cmd , const uint32_t width , const uint32_t height , const uint32_t depth = 1) const noexcept{
            if(!cmd || !table || !cmdTraceRaysKHR) return;
            cmdTraceRaysKHR(cmd , &rayGen , &miss , &hit , &callable , width , height , depth);
        }

    private:

        pGeneralBuffer2<BufferType::Device , false> table{nullptr};

        VkStridedDeviceAddressRegionKHR rayGen{};
        VkStridedDeviceAddressRegionKHR miss{};
        VkStridedDeviceAddressRegionKHR hit{};
        VkStridedDeviceAddressRegionKHR callable{};

        // read once ( the function static is initialised once , also when two threads come here at the same time )
        [[nodiscard]] static const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayTracingProperties(VulkanContext* context) noexcept{
            static const VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties = [context]{
                VkPhysicalDeviceRayTracingPipelinePropertiesKHR result{};
                result.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

                VkPhysicalDeviceProperties2 temp{};
                temp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                temp.pNext = &result;

                vkGetPhysicalDeviceProperties2(context->physicalDevice->device , &temp);
                result.pNext = nullptr;
                return result;
            }();
            return properties;
        }

        void processCopy(SBT2&& input) noexcept{
            table = std::move(input.table);
            rayGen = std::exchange(input.rayGen , {});
            miss = std::exchange(input.miss , {});
            hit = std::exchange(input.hit , {});
            callable = std::exchange(input.callable , {});
        }

        void processDelete() noexcept{
            table = nullptr;
            rayGen = {};
            miss = {};
            hit = {};
            callable = {};
        }

        [[nodiscard]] ErrorDataOutput<void> createSBT(const CreateInfo &input) noexcept{

            try{
                const auto& properties = rayTracingProperties(input.context);

                const uint32_t handleSize = properties.shaderGroupHandleSize;
                const uint32_t handleAlignment = properties.shaderGroupHandleAlignment;
                const uint32_t baseAlignment = properties.shaderGroupBaseAlignment;

                if(handleSize == 0 || handleAlignment == 0 || baseAlignment == 0){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create SBT2 , since the shader group handle size is 0 , is VK_KHR_ray_tracing_pipeline enabled?"});
                }

                const uint32_t groupCount = static_cast<uint32_t>(input.groupTypes.size());

                // the handles of all the groups , in the order of the pipeline
                std::vector<uint8_t> groupHandles(static_cast<size_t>(groupCount) * handleSize);
                {
                    const auto result = getRayTracingShaderGroupHandlesKHR(input.context->logicalDevice->device , input.pipeline , 0 , groupCount , static_cast<uint32_t>(groupHandles.size()) , groupHandles.data());
                    if(result != VK_SUCCESS){
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to get the ray tracing shader group handles for SBT2 , error ID -> {}" , static_cast<int32_t>(result))});
                    }
                }

                uint32_t rayGenCount = 0 , missCount = 0 , hitCount = 0 , callableCount = 0;
                for(const auto type : input.groupTypes){
                    switch(type){
                        case SBTGroupType::RayGeneration : rayGenCount++; break;
                        case SBTGroupType::Miss : missCount++; break;
                        case SBTGroupType::Hit : hitCount++; break;
                        case SBTGroupType::Callable : callableCount++; break;
                    }
                }

                if(rayGenCount != 1){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create SBT2 , exactly one ray generation group is required , got -> {}" , rayGenCount)});
                }

                // the stride of the records : the handle is aligned to shaderGroupHandleAlignment
                const uint32_t handleStride = alignUp(handleSize , handleAlignment);
                if(handleStride > properties.maxShaderGroupStride){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create SBT2 , the stride of a record -> {} is bigger than maxShaderGroupStride -> {}" , handleStride , properties.maxShaderGroupStride)});
                }

                // the ray generation region has one record , and its size has to be equal to its stride
                const uint32_t rayGenSize = alignUp(handleStride , baseAlignment);
                // every other region starts at an aligned offset and is as big as its records , rounded up to the alignment
                const uint32_t missSize = alignUp(handleStride * missCount , baseAlignment);
                const uint32_t hitSize = alignUp(handleStride * hitCount , baseAlignment);
                const uint32_t callableSize = alignUp(handleStride * callableCount , baseAlignment);

                const uint32_t missOffset = rayGenSize;
                const uint32_t hitOffset = missOffset + missSize;
                const uint32_t callableOffset = hitOffset + hitSize;
                const uint32_t totalSize = callableOffset + callableSize;

                // the table on the CPU : the handles are placed by the TYPE of the group , in the order they have in the pipeline
                std::vector<uint8_t> bytes(totalSize , 0);
                uint32_t rayGenAt = 0 , missAt = missOffset , hitAt = hitOffset , callableAt = callableOffset;
                for(uint32_t group = 0 ; group < groupCount ; group++){
                    const uint8_t* handle = groupHandles.data() + static_cast<size_t>(handleSize) * group;
                    switch(input.groupTypes[group]){
                        case SBTGroupType::RayGeneration :
                            std::memcpy(bytes.data() + rayGenAt , handle , handleSize);
                            break;
                        case SBTGroupType::Miss :
                            std::memcpy(bytes.data() + missAt , handle , handleSize);
                            missAt += handleStride;
                            break;
                        case SBTGroupType::Hit :
                            std::memcpy(bytes.data() + hitAt , handle , handleSize);
                            hitAt += handleStride;
                            break;
                        case SBTGroupType::Callable :
                            std::memcpy(bytes.data() + callableAt , handle , handleSize);
                            callableAt += handleStride;
                            break;
                    }
                }

                // the buffer ( the upload is done by GeneralBuffer2 : a staging buffer and a copy that is waited for )
                {
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = input.context;
                    info.size = totalSize;
                    info.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    table = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , translateToSpan(bytes));
                }

                const VkDeviceAddress base = table->pBuffer;
                if(base == 0 || (base % baseAlignment) != 0){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create SBT2 , the address of the table -> {} is not aligned to shaderGroupBaseAlignment -> {}" , base , baseAlignment)});
                }

                rayGen.deviceAddress = base;
                rayGen.stride = rayGenSize;
                rayGen.size = rayGenSize;

                // a region that has no groups stays empty ( address 0 , size 0 ) : nothing is read from it
                if(missCount){
                    miss.deviceAddress = base + missOffset;
                    miss.stride = handleStride;
                    miss.size = missSize;
                }
                if(hitCount){
                    hit.deviceAddress = base + hitOffset;
                    hit.stride = handleStride;
                    hit.size = hitSize;
                }
                if(callableCount){
                    callable.deviceAddress = base + callableOffset;
                    callable.stride = handleStride;
                    callable.size = callableSize;
                }
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the buffer of SBT2 -> {}" , e.what())});
            }

            return {};
        }
    };

    using pSBT2 = std::shared_ptr<SBT2>;
}
