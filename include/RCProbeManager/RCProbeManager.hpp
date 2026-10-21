#pragma once

#include "../Image2/RenderImage2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../Buffer2/GeneralBuffer.hpp"
#include <algorithm>
#include <random>
#include <ranges>

namespace mox{


    class RCProbeManager final{
    public:

        SetTypicalOperators(RCProbeManager)
        SetTypicalTransferOperators(RCProbeManager)

        void updatePoses(VkCommandBuffer& cmd , const glm::vec3 playerPos) noexcept{
            for(auto&& [readProbe , probe] : std::views::zip(generalProbesOffsets , generalProbes)){
                probe = readProbe;
                probe.position = readProbe.position + playerPos;
            }
            probesBuffer->updateBufferCmd(cmd , translateToSpan(generalProbes) , VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        }

        pDescriptorBuffer2 buffer{};
        pDescriptorLayout layout{};
        pGeneralBuffer2<BufferType::Device , true> probesBuffer{nullptr};
        // will be fixed , since it contains only ray cast and resolution of cascade
        pGeneralBuffer2<BufferType::Device , false> cascadeBuffer{nullptr};
        // probably fixed too 
        pGeneralBuffer2<BufferType::Device , false> generalProbeBuffer{nullptr};

        std::vector<pRenderImage2> cascades{};

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

        void updatePosesCreation(const glm::vec3 playerPos) noexcept{
            generalProbes.resize(generalProbesOffsets.size());
            for(auto&& [readProbe , probe] : std::views::zip(generalProbesOffsets , generalProbes)){
                probe = readProbe;
                probe.position = readProbe.position + playerPos;
            }
            probesBuffer->recordData(translateToSpan(generalProbes));
        }

        // general idea of this , is to make all offsets of probes , rather than allocating everytime new memory
        // since the amount of probes are not fixed between cascades , so we cant make 
        std::vector<RC::ProbeInfo> generalProbesOffsets;

        std::vector<RC::ProbeInfo> generalProbes;
        
        void createProbes(const glm::u16vec2 screenResolution , const uint32_t cascade , RC::CascadeInfo& cascadeInfo) noexcept {
            // so we will be making here all probes position per cascade
            // since we know our resolution of screen 
            // cascade is the ID , bigger cascade , lower the resolution and lower the count
            cascadeInfo.PROBES_PER_CASCADE = 0;
            cascadeInfo.PROBE_OFFSET = static_cast<uint32_t>(generalProbesOffsets.size());
            // the probes of a cascade lie on a SPHERE around the player : position is an OFFSET from the player ( updatePoses adds playerPos )
            static const float generalOffsetPerCascade = 10.0f; // the radius of the sphere of the cascade 0 , every next cascade is this much farther
            const float radius = generalOffsetPerCascade * static_cast<float>(cascade + 1u);

            // required radius , is radius for this cascade to be chosen
            cascadeInfo.requiredRadius2 = radius * radius;

            static const uint32_t generalHeadResolution = 64;
            // the bigger the cascade , the bigger the tile and the FEWER the probes
            const uint32_t probeResolution = generalHeadResolution * (cascade + 1u);
            // at least one probe per axis : a big cascade on a small screen must not be empty
            const uint32_t totalProbes_x = std::max(1u , (static_cast<uint32_t>(screenResolution.x) - 1u) / probeResolution);
            const uint32_t totalProbes_y = std::max(1u , (static_cast<uint32_t>(screenResolution.y) - 1u) / probeResolution);

            cascadeInfo.PROBE_RESOLUTION = probeResolution;
            cascadeInfo.PROBE_RAY_COUNT = 4u * (cascade + 1ul);

            cascadeInfo.CASCADE_X = totalProbes_x * probeResolution;
            cascadeInfo.CASCADE_Y = totalProbes_y * probeResolution;

            // the same seed for a cascade every launch : the offsets are made once , the picture must not change from run to run
            std::mt19937 engine(0x5EEDu + cascade);
            // the jitter moves a probe along the radius , by a share of it ( it was a fixed +-5 / ( cascade + 1 ) that got smaller with the cascade , while the sphere grows )
            const float jitter = 0.1f * radius;
            std::uniform_real_distribution<float> randomOffset(-jitter , jitter);

            generalProbesOffsets.reserve(generalProbesOffsets.size() + static_cast<size_t>(totalProbes_x) * totalProbes_y);

            for(uint32_t y = 0 ; y < totalProbes_y ; y++)
            for(uint32_t x = 0 ; x < totalProbes_x ; x++){
                // azimuth 0 .. 2pi around the up axis , the polar angle is taken through its COSINE ( from 1 at the top to -1 at the bottom , in equal steps ) :
                // the rings have the same area then , and the probes do not crowd at the poles like they do with equal steps of the angle
                const float azimuth = 2.0f * glm::pi<float>() * ((float)x + 0.5f) / (float)totalProbes_x;
                const float cosPolar = 1.0f - 2.0f * ((float)y + 0.5f) / (float)totalProbes_y;
                const float sinPolar = glm::sqrt(glm::max(0.0f , 1.0f - cosPolar * cosPolar));

                const glm::vec3 direction(sinPolar * glm::cos(azimuth) , cosPolar , sinPolar * glm::sin(azimuth));
 
                RC::ProbeInfo probe{};
                probe.validProbe = true;
                probe.cascadeID = cascade;
                probe.position = direction * (radius + randomOffset(engine));
                generalProbesOffsets.push_back(probe);
            }

            cascadeInfo.PROBES_PER_CASCADE = totalProbes_x * totalProbes_y;
        }

        [[nodiscard]] ErrorDataOutput<void> createBuffersAndImages(const glm::u16vec2 screenResolution, const glm::vec3 playerPos) noexcept{
            try{
                static const uint32_t cascadeCount = 5;
                std::vector<RC::CascadeInfo> cascadesInfos;

                for(uint32_t i = 0 ; i < cascadeCount ; i++){
                    RC::CascadeInfo cascade{};
                    createProbes(screenResolution , i , cascade);
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
                    
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(RC::GeneralProbeInfo);
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

                    generalProbeBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , &data);
                }

                {
                    // cascades
                    
                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(RC::CascadeInfo) * cascadesInfos.size();
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

                    cascadeBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , translateToSpan(cascadesInfos));
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
                    std::vector<DescriptorLayoutData> data(3);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    for(auto& cascade : cascades){
                        const uint32_t binding = data.size();
                        DescriptorLayoutData input = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                        data.push_back(input);
                    }

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

                    for(auto& cascade : cascades){
                        const uint32_t binding = data.size();
                        DescriptorData2 input = DescriptorData2::createData(cascade.get(), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , binding , 0);
                        data.push_back(input);
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
