#pragma once

#include "../Image2/RenderImage2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../Buffer2/GeneralBuffer.hpp"
#include <algorithm>
#include <cmath>

namespace mox{

    // =============================== INFO ===============================
    // RCProbeManager - the sources of Radiance Cascade with the probes of the SCREEN : only the creation , nothing is managed after it
    //
    // a cascade covers the WHOLE screen , the cascades differ in how dense the probes are ( that is the hierarchy ) :
    //   the tile of a probe  : BASE_PROBE_RESOLUTION * 2 ^ cascade pixels  ( the next cascade has 4 times less probes )
    //   the block of a probe : BUILD_SIZE x BUILD_SIZE texels ( one texel = one ray , the octahedral map of the directions ) , BASE_BUILD_SIZE + 2 * cascade
    //   the image of a cascade : ( probes on x * BUILD_SIZE ) x ( probes on y * BUILD_SIZE ) , the block of the probe ( x , y ) is at ( x , y ) * BUILD_SIZE
    // the probes of all the cascades are in ONE buffer : the cascade starts at PROBE_OFFSET , the probe ( x , y ) is  PROBE_OFFSET + y * PROBE_PER_ROW + x
    // the positions are written every frame by the placement pass ( RCplacement.hlsl ) from the G buffer , so nothing is filled here
    //
    // the descriptor set ( the same as in the shaders : RCProbeBuffer , RCCascadesBuffer , RCProbeInfo , RCImages[] ) :
    //   0 : the probes , 1 : the cascades , 2 : the general info ( uniform ) , 3 : the images of the cascades ( ONE binding , an array , the cascade is the index )
    // the size of the screen is the one at the creation : when the window is resized the manager has to be made again
    // =============================== END ===============================

    class RCProbeManager final{
    public:

        SetTypicalOperators(RCProbeManager)
        SetTypicalTransferOperators(RCProbeManager)

        static constexpr uint32_t CASCADE_COUNT = 5;
        // the tile of a probe of the cascade 0 , in pixels
        static constexpr uint32_t BASE_PROBE_RESOLUTION = 16;
        // the block of a probe of the cascade 0 : BASE_BUILD_SIZE ^ 2 rays
        static constexpr uint32_t BASE_BUILD_SIZE = 4;
        // the end of the interval of the cascade 0 , meters : the end of a cascade is the start of the next one , and it is twice bigger
        static constexpr float BASE_INTERVAL = 6.0f;
        // the end of the interval of the last cascade
        static constexpr float LAST_INTERVAL_END = 1000.0f;

        pDescriptorBuffer2 buffer{};
        pDescriptorLayout layout{};
        pGeneralBuffer2<BufferType::Device , true> probesBuffer{nullptr};
        // written once : the sizes of the images , the offsets of the probes , the intervals
        pGeneralBuffer2<BufferType::Device , true> cascadeBuffer{nullptr};
        // written once : the number of the cascades and of the probes
        pGeneralBuffer2<BufferType::Device , true> generalProbeBuffer{nullptr};

        std::vector<pRenderImage2> cascades{};
        // the same data as in the cascadeBuffer : the passes use it to dispatch ( how many probes , how many texels ) without reading the buffer back
        std::array<RC::CascadeInfo , CASCADE_COUNT> cascadesData{};
        uint32_t totalProbes{0};

        struct CreateInfo{
            VulkanContext* context;
        };

        explicit RCProbeManager(const CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeQueueManager | VkReqTypeLogicalDevice | VkReqTypePhysicalDevice , info.context);
                if(!result.has_value()){
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            this->context = info.context;
            {
                const auto result = __implInit();
                if(!result.has_value()){
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }
        }

        ~RCProbeManager() noexcept{
            processDelete();
        }

    private:

        VulkanContext* context{nullptr};

        void processCopy(RCProbeManager&& input) noexcept{
            buffer = std::move(input.buffer);
            layout = std::move(input.layout);
            probesBuffer = std::move(input.probesBuffer);
            generalProbeBuffer = std::move(input.generalProbeBuffer);
            cascadeBuffer = std::move(input.cascadeBuffer);
            cascades = std::move(input.cascades);
            cascadesData = input.cascadesData;
            totalProbes = std::exchange(input.totalProbes , 0u);
            context = std::exchange(input.context , nullptr);
        }

        void processDelete() noexcept{
            buffer = nullptr;
            layout = nullptr;
            probesBuffer = nullptr;
            cascadeBuffer = nullptr;
            generalProbeBuffer = nullptr;
            cascades.clear();
            totalProbes = 0;
            context = nullptr;
        }

        // the info of one cascade ( the offset of its probes is set by the caller , it depends on the cascades before )
        void __implProcessCascade(const uint32_t id , RC::CascadeInfo &cs) noexcept
        {
            const uint32_t screenWidth = context->sc->extent.width;
            const uint32_t screenHeight = context->sc->extent.height;

            const uint32_t tile = BASE_PROBE_RESOLUTION << id;
            const uint32_t buildSize = BASE_BUILD_SIZE + 2u * id;

            // the last tile is cut by the edge of the screen , but a probe is there : the number of the probes is rounded up
            const uint32_t probesX = (screenWidth + tile - 1u) / tile;
            const uint32_t probesY = (screenHeight + tile - 1u) / tile;

            cs = {};
            cs.PROBE_RESOLUTION = static_cast<float>(tile);
            cs.BUILD_SIZE = buildSize;
            cs.BLOCK_SIZE = buildSize;
            cs.PROBE_RAY_COUNT = buildSize * buildSize;
            cs.PROBE_PER_ROW = probesX;
            cs.PROBES_PER_CASCADE = probesX * probesY;
            cs.CASCADE_X = probesX * buildSize;
            cs.CASCADE_Y = probesY * buildSize;
            cs.CELL_SIZE = 0.0f;

            // the interval of the rays : [ 0 , BASE ] , [ BASE , 2 BASE ] , [ 2 BASE , 4 BASE ] ... the last one goes on to the end
            const bool last = (id + 1u == CASCADE_COUNT);
            cs.INTERVAL_START = (id == 0u) ? 0.0f : BASE_INTERVAL * static_cast<float>(1u << (id - 1u));
            cs.INTERVAL_END = last ? LAST_INTERVAL_END : BASE_INTERVAL * static_cast<float>(1u << id);
        }

        [[nodiscard]] ErrorDataOutput<void> __implInit() noexcept
        {
            try{
                // the info of the cascades and the offsets of their probes in the ONE buffer of the probes
                uint32_t offset = 0;
                for(uint32_t i = 0 ; i < CASCADE_COUNT ; i++){
                    __implProcessCascade(i , cascadesData[i]);
                    cascadesData[i].PROBE_OFFSET = offset;
                    offset += cascadesData[i].PROBES_PER_CASCADE;
                }
                totalProbes = offset;

                // an image for every cascade
                for(uint32_t i = 0 ; i < CASCADE_COUNT ; i++){
                    RenderImage2::CreateInfo info{};
                    info.context = this->context;
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.extent = {cascadesData[i].CASCADE_X , cascadesData[i].CASCADE_Y};
                    info.mipMapCount = 1;
                    info.layerCount = 1;
                    info.clear.float32[0] = 0.0f;
                    info.clear.float32[1] = 0.0f;
                    info.clear.float32[2] = 0.0f;
                    info.clear.float32[3] = 0.0f;
                    cascades.push_back(std::make_shared<RenderImage2>(info));
                }

                {
                    // the general data
                    RC::GeneralProbeInfo data{};
                    data.TOTAL_PROBES = totalProbes;
                    data.TOTAL_CASCADE_COUNT = CASCADE_COUNT;

                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(RC::GeneralProbeInfo);
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

                    generalProbeBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info , &data);
                }

                {
                    // the cascades
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(RC::CascadeInfo) * CASCADE_COUNT;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

                    cascadeBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info , translateToSpan(cascadesData.data() , cascadesData.size()));
                }

                {
                    // the probes : written by the placement pass every frame
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(RC::ProbeInfo) * totalProbes;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

                    probesBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }

                {
                    std::vector<DescriptorLayoutData> data(4);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , CASCADE_COUNT};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    layout = std::make_shared<DescriptorLayout>(info);
                }

                {
                    std::vector<DescriptorData2> data{};
                    data.push_back(DescriptorData2::createData(probesBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0));
                    data.push_back(DescriptorData2::createData(cascadeBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0));
                    data.push_back(DescriptorData2::createData(generalProbeBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0));

                    // ONE binding ( 3 ) for all the cascades , as in the layout above ( count = CASCADE_COUNT ) and as in the shaders ( RCImages[] ) :
                    // the cascade is the index in the array of the binding , not a binding of its own
                    for(uint32_t i = 0 ; i < cascades.size() ; i++){
                        data.push_back(DescriptorData2::createData(cascades[i].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , i));
                    }

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = layout.get();

                    buffer = std::make_shared<DescriptorBuffer2>(info);
                }
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create sources for RCProbeManager"});
            }

            return {};
        }

    };

    using pRCProbeManager = std::shared_ptr<RCProbeManager>;
}
