#pragma once

#include "GeneralBuffer.hpp"


namespace mox{

    class SourceBuffers final{
    public:

        SetTypicalOperators(SourceBuffers)
        // both are default , since we got the shared ptrs
        SourceBuffers& operator=(SourceBuffers&& input) = default;
        SourceBuffers(SourceBuffers&& input) = default;

        explicit SourceBuffers(VulkanContext* context){
            const auto result = createAllBuffers(context);
            if(!result.has_value()){
                processDelete();
                engineLogger(result);
                THROW_MESSAGE;
            }
        }

        pDeviceReadableBuffer generalBuffer{nullptr};
        pDeviceReadableBuffer features{nullptr};
        pDeviceReadableBuffer sunBuffer{nullptr};
        pDeviceReadableBuffer luminanceHistory{nullptr};
        // FogVolume : read by fogCalc / PostProcess , edited by the ImGuiPipe ( the CPU reads it )
        pDeviceReadableBuffer fogData{nullptr};

        pGeneralBuffer2<BufferType::Device , true> forwardInfo{nullptr};
        // since we dont really care about reading it 

        pGeneralBuffer2<BufferType::Device , false> tileLightIndices{nullptr};
        pGeneralBuffer2<BufferType::Device , false> lightCountPerTile{nullptr};

        pGeneralBuffer2<BufferType::Device , false> ReSTIRDIBuffer[2] {nullptr , nullptr};
        pGeneralBuffer2<BufferType::Device , false> ReSTIRGIBuffer[2] {nullptr , nullptr};
        // the reservoirs of ReSTIR PT ( ReservoirPT , one per pixel ) : ping pong like the others , this frame writes its own and reads the other
        pGeneralBuffer2<BufferType::Device , false> ReSTIRPTBuffer[2] {nullptr , nullptr};

        pGeneralBuffer2<BufferType::Device , true> skyBuffer{nullptr};

        ~SourceBuffers(){
            processDelete();
        }

    private:

        void processDelete() noexcept{
            generalBuffer = nullptr;
            features = nullptr;
            sunBuffer = nullptr;
            luminanceHistory = nullptr;
            fogData = nullptr;
            skyBuffer = nullptr;
            ReSTIRDIBuffer[0] = nullptr;
            ReSTIRDIBuffer[1] = nullptr;
            ReSTIRGIBuffer[0] = nullptr;
            ReSTIRGIBuffer[1] = nullptr;
            ReSTIRPTBuffer[0] = nullptr;
            ReSTIRPTBuffer[1] = nullptr;
            
            tileLightIndices = nullptr;
            lightCountPerTile = nullptr;
            forwardInfo = nullptr;
        }

        [[nodiscard]] ErrorDataOutput<void> createAllBuffers(VulkanContext* context) noexcept{

            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeInstance | VkReqTypePhysicalDevice , context);
                if(!result.has_value()) return result;
            }
            try{

                GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                info.context= context;
                info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                
                // general buffer
                info.size = sizeof(GeneralInfo);
                generalBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                // features
                info.size = sizeof(GeneralFeatures);
                features = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                // sun buffer
                info.size = sizeof(SunSource);
                sunBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                // luminance history
                info.size = sizeof(uint32_t) * 257;
                luminanceHistory = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                info.size = sizeof(SkyViewData);
                skyBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                info.size = sizeof(FogVolume);
                fogData = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                ForwardPlusInfo data{};
                data.TILE_SIZE = 16;

                uint32_t tilesX = (context->sc->extent.width  + data.TILE_SIZE - 1) / data.TILE_SIZE;
                uint32_t tilesY = (context->sc->extent.height + data.TILE_SIZE - 1) / data.TILE_SIZE;

                data.TILE_PER_ROW      = tilesX;
                data.TILE_TOTAL_AMOUNT = tilesX * tilesY;
                data.totalLights       = 0;

                info.size = sizeof(ForwardPlusInfo);
                forwardInfo = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info , &data);

                for(auto& id : {0 , 1}){
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(ReservoirDI) * context->sc->extent.width * context->sc->extent.height;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    ReservoirDI di{};
                    di.chosenIndex = 9999;
                    
                    di.M = 0;
                    di.W = 0.0f;
                    di.weightSum = 0.0f;
                    std::vector<ReservoirDI> data(context->sc->extent.width * context->sc->extent.height , di );
                    // the reservoirs START EMPTY ( M = 0 , W = 0 ) : the data was made above but never given to the buffer , and a device buffer is not zeroed by itself .
                    // the garbage of the memory ( huge or NaN W ) was read by the temporal reuse of the pixel and , worse , by the spatial reuse of its neighbours , who
                    // copied it and kept it for ever
                    ReSTIRDIBuffer[id] = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , static_cast<void*>(data.data()));
                    info.size = sizeof(ReservoirGI) * context->sc->extent.width * context->sc->extent.height;
                    const std::vector<ReservoirGI> emptyGI(static_cast<size_t>(context->sc->extent.width) * context->sc->extent.height , ReservoirGI{});
                    ReSTIRGIBuffer[id] = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , const_cast<void*>(static_cast<const void*>(emptyGI.data())));

                    // the reservoirs of the path tracer start empty too ( M = 0 , W = 0 ) : a device buffer is not zeroed by itself
                    info.size = sizeof(ReservoirPT) * context->sc->extent.width * context->sc->extent.height;
                    const std::vector<ReservoirPT> emptyPT(static_cast<size_t>(context->sc->extent.width) * context->sc->extent.height , ReservoirPT{});
                    ReSTIRPTBuffer[id] = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , const_cast<void*>(static_cast<const void*>(emptyPT.data())));
                }

                {   


                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context= context;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.size = sizeof(uint32_t) * MAX_LIGHT_SOURCE * data.TILE_TOTAL_AMOUNT;

                    tileLightIndices = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info);

                    info.size = sizeof(uint32_t) * data.TILE_TOTAL_AMOUNT;

                    lightCountPerTile = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION, "failed to create typical source buffers"});
            }

            return {};
        }
    };

    using pSourceBuffers = std::shared_ptr<SourceBuffers>;
}
