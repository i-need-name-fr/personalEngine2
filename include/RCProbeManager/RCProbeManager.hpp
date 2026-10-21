#pragma once

#include "../Image2/RenderImage2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../Buffer2/GeneralBuffer.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <ranges>

namespace mox{


    class RCProbeManager final{
    public:

        SetTypicalOperators(RCProbeManager)
        SetTypicalTransferOperators(RCProbeManager)

        // the grid of every cascade : GRID_SIZE nodes on an axis , the cell of the cascade 0 is BASE_CELL_SIZE meters , every next cascade is twice bigger
        static constexpr uint32_t GRID_SIZE = 32;
        static constexpr float BASE_CELL_SIZE = 1.5f;

        void updatePoses(VkCommandBuffer& cmd , const glm::vec3 playerPos) noexcept{
            computeProbePositions(playerPos);
            probesBuffer->updateBufferCmd(cmd , translateToSpan(generalProbes) , VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        }

        pDescriptorBuffer2 buffer{};
        pDescriptorLayout layout{};
        pGeneralBuffer2<BufferType::Device , true> probesBuffer{nullptr};
        // will be fixed , since it contains only ray cast and resolution of cascade
        pGeneralBuffer2<BufferType::Device , true> cascadeBuffer{nullptr};
        // probably fixed too 
        pGeneralBuffer2<BufferType::Device , true> generalProbeBuffer{nullptr};

        std::vector<pRenderImage2> cascades{};
        std::vector<RC::ProbeInfo> generalProbesOffsets;

        struct CreateInfo{
            VulkanContext* context;
            glm::vec3 playerPos;
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
                const auto result = createBuffersAndImages({info.context->sc->extent.width , info.context->sc->extent.height} , info.playerPos);
                if(!result.has_value()){
                    processDelete();
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }
        }

        ~RCProbeManager() noexcept{
            processDelete();
        }

    private:

        VulkanContext* context{nullptr};
        // we will change it per frame

        void processCopy(RCProbeManager&& input) noexcept{
            buffer = std::move(input.buffer);
            probesBuffer = std::move(input.probesBuffer);
            generalProbeBuffer = std::move(input.generalProbeBuffer);
            cascadeBuffer = std::move(input.cascadeBuffer);
            cascades = std::move(input.cascades);
            context = std::exchange(input.context,  nullptr);
            layout = std::move(input.layout);
            generalProbesOffsets = std::move(input.generalProbesOffsets);
            generalProbes = std::move(input.generalProbes);
            cellSizes = std::move(input.cellSizes);

        }

        void processDelete() noexcept{
            buffer = nullptr;
            context = nullptr;
            probesBuffer = nullptr;
            cascadeBuffer = nullptr;
            cascades.clear();
            generalProbeBuffer = nullptr;
            layout = nullptr;
        }

        // the world positions of the probes : the grid of a cascade is SNAPPED to its cell ( the centre is the player rounded down to the cell ) , so a probe stays at the
        // same place of the world while the player moves inside of a cell , and moves by a whole cell when he crosses its border . validProbe is 1 again for all the probes
        // ( the rays of the frame take it away from the probes that are inside of an object )
        void computeProbePositions(const glm::vec3 playerPos) noexcept{
            generalProbes.resize(generalProbesOffsets.size());
            for(auto&& [readProbe , probe] : std::views::zip(generalProbesOffsets , generalProbes)){
                const float cell = cellSizes[readProbe.cascadeID];
                const glm::vec3 snapped = glm::floor(playerPos / cell) * cell;
                probe = readProbe;
                probe.position = snapped + readProbe.position;
            }
        }

        void updatePosesCreation(const glm::vec3 playerPos) noexcept{
            computeProbePositions(playerPos);
            probesBuffer->recordData(translateToSpan(generalProbes));
        }

        // general idea of this , is to make all offsets of probes , rather than allocating everytime new memory
        // since the amount of probes are not fixed between cascades , so we cant make 

        std::vector<RC::ProbeInfo> generalProbes;
        
        // the size of the cell of every cascade ( the index is the cascade )
        std::vector<float> cellSizes;

        // the probes of a cascade : a 3D grid of GRID_SIZE ^ 3 nodes around the player . the offset of a node from the centre of the grid is stored ( the centre is the player
        // rounded down to the cell , see computeProbePositions ) . every cascade has the same number of probes , but its cell is twice bigger than the one before , so the grid
        // covers twice more space and sees farther ( with the rays that start where the rays of the cascade before end )
        void createProbes(const uint32_t cascade , const bool lastCascade , RC::CascadeInfo& cascadeInfo) noexcept {
            const uint32_t n = GRID_SIZE;
            const float cell = BASE_CELL_SIZE * static_cast<float>(1u << cascade);
            if(cellSizes.size() <= cascade) cellSizes.resize(cascade + 1u , 0.0f);
            cellSizes[cascade] = cell;

            const uint32_t total = n * n * n;
            cascadeInfo.PROBE_OFFSET = static_cast<uint32_t>(generalProbesOffsets.size());
            cascadeInfo.PROBES_PER_CASCADE = total;
            cascadeInfo.GRID_SIZE = n;
            cascadeInfo.CELL_SIZE = cell;

            // the rays : one for every texel of the block of a probe ( the ray x is the texel x , its direction is octahedralDecode of the texel ) . the bigger the cascade ,
            // the more directions : the farther the interval , the finer the angle has to be
            cascadeInfo.BLOCK_SIZE = 4u + 2u * cascade;
            cascadeInfo.PROBE_RAY_COUNT = cascadeInfo.BLOCK_SIZE * cascadeInfo.BLOCK_SIZE;

            // the interval of the rays : 4 cells of the cascade . the end of a cascade is the start of the next one , the last cascade goes on to infinity ( the sky )
            cascadeInfo.INTERVAL_START = (cascade == 0u) ? 0.0f : 4.0f * BASE_CELL_SIZE * static_cast<float>(1u << (cascade - 1u));
            cascadeInfo.INTERVAL_END = lastCascade ? 1000.0f : 4.0f * cell;

            // the image of the cascade : the blocks of the probes in rows ( the block of the probe p is at ( p % row , p / row ) * BLOCK_SIZE )
            const uint32_t row = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(total))));
            const uint32_t rows = (total + row - 1u) / row;
            cascadeInfo.CASCADE_X = row * cascadeInfo.BLOCK_SIZE;
            cascadeInfo.CASCADE_Y = rows * cascadeInfo.BLOCK_SIZE;

            generalProbesOffsets.reserve(generalProbesOffsets.size() + total);

            const float half = static_cast<float>(n / 2u);
            for(uint32_t z = 0 ; z < n ; z++)
            for(uint32_t y = 0 ; y < n ; y++)
            for(uint32_t x = 0 ; x < n ; x++){
                RC::ProbeInfo probe{};
                probe.validProbe = true;
                probe.cascadeID = cascade;
                probe.position = glm::vec3(static_cast<float>(x) - half , static_cast<float>(y) - half , static_cast<float>(z) - half) * cell;
                generalProbesOffsets.push_back(probe);
            }
        }

        [[nodiscard]] ErrorDataOutput<void> createBuffersAndImages(const glm::u16vec2 screenResolution, const glm::vec3 playerPos) noexcept{
            try{
                static const uint32_t cascadeCount = 5;
                std::vector<RC::CascadeInfo> cascadesInfos;

                for(uint32_t i = 0 ; i < cascadeCount ; i++){
                    RC::CascadeInfo cascade{};
                    createProbes(i , i + 1u == cascadeCount , cascade);
                    cascadesInfos.push_back(cascade);
                    // now we need to create image for that cascade
                    {
                        RenderImage2::CreateInfo info{};
                        info.context = this->context;
                        info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                        info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                        info.imageType = VK_IMAGE_TYPE_2D;
                        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                        info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                        info.extent = {cascade.CASCADE_X , cascade.CASCADE_Y};
                        info.mipMapCount = 1;
                        info.layerCount = 1;
                        info.clear.float32[0] = 0.0f;
                        info.clear.float32[1] = 0.0f;
                        info.clear.float32[2] = 0.0f;
                        info.clear.float32[3] = 0.0f;
                        cascades.push_back(std::move(std::make_shared<RenderImage2>(info)));
                    }
                }

                // now creating all the buffers
                {
                    // general data
                    RC::GeneralProbeInfo data{};
                    data.TOTAL_PROBES = generalProbesOffsets.size();
                    data.TOTAL_CASCADE_COUNT = cascadeCount;
                    
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(RC::GeneralProbeInfo);
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

                    generalProbeBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info , &data);
                }

                {
                    // cascades
                    
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(RC::CascadeInfo) * cascadesInfos.size();
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

                    cascadeBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info , translateToSpan(cascadesInfos));
                }

                {
                    // probes
                    
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(RC::ProbeInfo) * generalProbesOffsets.size();
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

                    probesBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }
                updatePosesCreation(playerPos);

                {
                    std::vector<DescriptorLayoutData> data(4);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , cascadeCount};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    layout = std::make_shared<DescriptorLayout>(info);
                }

                {
                    // now we need to createDescriptor buffer for this
                    // 3 for buffers and other N for 
                    std::vector<DescriptorData2> data(3);
                    data[0] = DescriptorData2::createData(probesBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0 );
                    data[1] = DescriptorData2::createData(cascadeBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0);
                    data[2] = DescriptorData2::createData(generalProbeBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0);

                    // ONE binding ( 3 ) for all the cascades , as in the layout above ( count = cascadeCount ) and as in the shaders ( RCImages[] ) :
                    // the cascade is the index in the array of the binding , not a binding of its own ( the bindings 3 , 4 , 5 ... do not exist in the layout )
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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION  , "failed to create sources for RCProbeManager"});
            }

            return{};
        }
    };

    using pRCProbeManager = std::shared_ptr<RCProbeManager>;
}
