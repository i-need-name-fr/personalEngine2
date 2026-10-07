#pragma once

#include "buffer.hpp"


namespace mox{


    enum class RayTracingShaderType{
        ClosestHit,
        Miss,
        RayGeneration,
        AnyHit // i will probably add this later to the list
    };

    class SBT final : public BufferBase{
    public:

        SBT() = delete;
        SBT(SBT& input) = delete;
        SBT(const SBT& input) = delete;
        SBT(SBT&& input) noexcept : BufferBase(std::move(input)) {
            regions = std::move(input.regions);
        }

        SBT& operator=(SBT& input) = delete;
        SBT& operator=(const SBT& input) = delete;
        SBT& operator=(SBT&& input) noexcept {
            performCopy(std::move(input));
            regions = std::move(input.regions);

            return *this;
        }

        struct CreateInfo{

            VulkanContext* context;

            VkPipeline pipeline;
            // GROUP COUNT SHOULD BE EQUAL TO SHADER TYPES SIZE
            // shaderTypes[i] is the type of the pipeline's group i ( exactly one RayGeneration ).
            // inside Miss / ClosestHit the groups keep their relative order , that order is
            // what the miss index / hit group index in TraceRay refers to
            std::vector<RayTracingShaderType> shaderTypes;
        };


        SBT(const CreateInfo &info) : BufferBase(info.context){

            if(!info.pipeline){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION ,"failed to create SBT , since the pipeline is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(info.shaderTypes.empty()){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION ,"failed to create SBT , since the shader types are empty"};
                engineLogger(&error);
                THROW_MESSAGE;
            }


            {
                const auto result = createSBT(info);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    cleanBuffer();
                    regions ={};
                    THROW_MESSAGE;
                }
            }
        }

        void execute(VkCommandBuffer &cmd , const uint32_t width , const uint32_t height) noexcept{
            if(!cmd) return;
            if(regions.empty()) return;
            cmdTraceRaysKHR(cmd , &regions[0] , &regions[1] , &regions[2] , &regions[3] , width ,height , 1);
        }

        ~SBT(){
            regions = {};
        }

    private:

        std::vector<VkStridedDeviceAddressRegionKHR> regions{};
        inline static VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{};
        inline static uint8_t hasLoadedProperties;

        [[nodiscard]] ErrorDataOutput<void> createSBT(const CreateInfo &input) noexcept{

            try{
                if(!hasLoadedProperties){
                    properties = {};
                    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

                    VkPhysicalDeviceProperties2 temp{};
                    temp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                    temp.pNext = &properties;

                    vkGetPhysicalDeviceProperties2(input.context->physicalDevice->device , &temp);
                    hasLoadedProperties = true;
                }

                const uint32_t handleSize = properties.shaderGroupHandleSize;
                const uint32_t handleAlignment = properties.shaderGroupHandleAlignment;
                const uint32_t baseAlignment = properties.shaderGroupBaseAlignment;

                if(handleSize == 0){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create SBT , since the shader group handle size is 0 , is VK_KHR_ray_tracing_pipeline enabled?"});
                }

                VkDeviceSize totalSize = static_cast<VkDeviceSize>(input.shaderTypes.size()) * handleSize;
                std::vector<uint8_t> groupHandle(totalSize);

                {
                    const auto result = getRayTracingShaderGroupHandlesKHR(input.context->logicalDevice->device , input.pipeline , 0 , input.shaderTypes.size() , static_cast<uint32_t>(groupHandle.size()) , groupHandle.data());
                    if(result != VK_SUCCESS){
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to get the ray tracing shader group handles for SBT , error ID -> {}" , static_cast<int32_t>(result))});
                    }
                }

                uint32_t rayGenCount = 0;
                uint32_t missCount = 0;
                uint32_t hitCount = 0;

                for(auto& data :input.shaderTypes){
                    if(data == RayTracingShaderType::RayGeneration){
                        rayGenCount++;
                    }else if(data == RayTracingShaderType::ClosestHit){
                        hitCount++;
                    }else if(data == RayTracingShaderType::Miss){
                        missCount++;
                    }else{
                        // AnyHit used to be silently skipped here while still shifting every
                        // group index after it
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create SBT , since AnyHit groups arent supported yet"});
                    }
                }

                if(rayGenCount != 1){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create SBT , exactly one ray generation group is required , got -> {}" , rayGenCount)});
                }

                const uint32_t rayGenStrideSize = alignUp(alignUp(handleSize ,handleAlignment ) , baseAlignment);
                const uint32_t handleStride =alignUp(handleSize , handleAlignment);

                const uint32_t hitSize = alignUp(handleStride * hitCount , baseAlignment);
                const uint32_t missSize = alignUp(handleStride * missCount , baseAlignment);

                pHostBuffer staging = nullptr;

                {
                    HostBuffer::CreateInfo info{};
                    info.context= input.context;
                    info.usage=  VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    info.size = rayGenStrideSize + missSize + hitSize;

                    staging = std::make_shared<HostBuffer>(info);
                }
                
                uint8_t* base = std::bit_cast<uint8_t*>(staging->data);
                memset(base , 0 , rayGenStrideSize + missSize + hitSize);

                // every region starts at its own aligned offset - the hit region used to start
                // right after the last miss handle , while regions[2] pointed at
                // rayGenStrideSize + missSize , so with an unaligned miss count the hit
                // handles were read from the wrong place
                uint8_t* rayGenPtr = base;
                uint8_t* missPtr = base + rayGenStrideSize;
                uint8_t* hitPtr = base + rayGenStrideSize + missSize;

                // groups are placed by their declared type , not by their position in the pipeline ,
                // inside one type the order of the groups is kept ( miss index 0 , 1 , 2 ... )
                for(uint32_t group = 0 ; group < input.shaderTypes.size() ; group++){
                    const uint8_t* handle = groupHandle.data() + static_cast<std::size_t>(handleSize) * group;

                    switch(input.shaderTypes[group]){
                        case RayTracingShaderType::RayGeneration :
                            memcpy(rayGenPtr , handle , handleSize);
                            break;
                        case RayTracingShaderType::Miss :
                            memcpy(missPtr , handle , handleSize);
                            missPtr += handleStride;
                            break;
                        case RayTracingShaderType::ClosestHit :
                            memcpy(hitPtr , handle , handleSize);
                            hitPtr += handleStride;
                            break;
                        default:
                            break;
                    }
                }

                // now the buffer base
                {
                    BufferBaseCreateInfo info{};
                    info.context = input.context;
                    info.memoryProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    info.size = rayGenStrideSize + missSize + hitSize;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

                    const auto result = createBuffer(info);
                    if(!result.has_value()) return std::unexpected(result.error());
                }

                // copyBufferToBuffer runs its own Begin/recordCmd/End on the queue manager , so it
                // must NOT be called from inside a recordCmd lambda - the nested End() re-ran the
                // outer lambda again and again ( endless recursion ) and push_back'ed into the
                // work vector that was being executed
                {
                    const auto result = copyBufferToBuffer(staging->buffer , this->buffer , this->bufferSize);
                    if(!result.has_value()) return std::unexpected(result.error());
                }

                regions.resize(4);

                VkDeviceAddress add = this->pBuffer;

                regions[0].deviceAddress = add;
                regions[0].size = rayGenStrideSize;
                regions[0].stride = rayGenStrideSize;

                regions[1].deviceAddress = add + rayGenStrideSize;
                regions[1].size = missSize;
                regions[1].stride = handleStride;

                regions[2].deviceAddress = add + rayGenStrideSize + missSize;
                regions[2].size = hitSize;
                regions[2].stride = handleStride;

                regions[3] = {};

                staging = nullptr;
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create SBT buffer"});
            }

            return {};
        }
    };


    using pSBT = std::shared_ptr<SBT>;
}