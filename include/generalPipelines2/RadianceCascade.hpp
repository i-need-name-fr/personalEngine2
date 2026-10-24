#pragma once

#include "../Pipeline/Pipeline.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../AS2/AccelerationStructureBase.hpp"
#include "../Object2/Mesh2.hpp"
#include "../Buffer2/SBT2.hpp"
#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../Sampler/Sampler.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../RCProbeManager/RCProbeManager.hpp"
#include "../LightManager/LightManager.hpp"

namespace mox{

    // =============================== INFO ===============================
    // RadianceCascade - Radiance Cascade with the probes of the SCREEN ( RCProbeManager ) : the placement of the probes , their rays and the merge of the cascades
    //
    // one frame of the pass ( after the deferred render : the G buffer of this frame is ready ) :
    //   1. the placement ( RCplacement.hlsl , compute ) : for every cascade a probe stands on the surface of the middle pixel of its tile : position , normal , valid or not
    //   2. the rays ( RCgeneration.hlsl and the others , ray tracing ) : for every cascade the grid of the rays is the IMAGE of the cascade , one ray = one texel
    //      ( the probe and the texel in its block come from the coordinates of the texel ) . the cascade is given to the ray generation shader by the push constant
    //   3. the merge from the top ( CombineCascades.hlsl , compute ) : ( prev = last , current = last - 1 ) ... ( prev = 1 , current = 0 ) , one dispatch for every pair
    // after it the cascade 0 holds the light of all the distances : the ray tracing pass reads only it ( RC::processRC in generalHit.hlsl )
    //
    // the descriptors of the rays , four sets :
    //   set 0 : 19 bindings of the scene ( the same list in RCgeneration / RChit / RCmiss / RCshadow* ; 8 is not used by the shaders ) : the acceleration structure ,
    //           the uniform buffers , the lamps of the tiles , the geometry and the materials , the sky
    //   set 1 : the textures of the TextureManager2
    //   set 2 : the set of the RCProbeManager : the probes , the cascades , the general data , RCImages[] ( ONE binding , an array of the cascade images )
    //   set 3 : the world grid of the lamps of the LightManager ( RChit.hlsl takes the lamps of the hit from it )
    // the placement has a set of its own ( the G buffer of the frame : two descriptor buffers , the parity of the frame ) , the merge uses only the set of the probes
    //
    // the order of the groups of the ray tracing pipeline :
    //   miss : 0 the sky ( RCmiss ) , 1 the shadow ( RCshadowMiss )
    //   hit  : 0 the closest hit ( RChit ) , 1 the shadow ( the any hit , and the closest hit when shadowHitPath is not empty )
    // =============================== END ===============================

    class RadianceCascade : public GraphPass2{
    public:

        SetTypicalOperators(RadianceCascade)
        SetTypicalTransferOperators(RadianceCascade)

        struct CreateInfo{
            VulkanContext* context{};
            AccelerationStructure* as{};
            TextureManager2* manager{};
            RCProbeManager* probes{};
            LightManager* lights{};

            std::string placementPath;      // the compute shader that puts the probes on the surfaces
            std::string rayGenerationPath;
            std::string missPath;           // miss 0
            std::string shadowMissPath;     // miss 1
            std::string hitPath;            // hit group 0 : the closest hit
            std::string shadowAnyHitPath;   // hit group 1 : the any hit
            std::string shadowHitPath;      // hit group 1 , optional : the closest hit ( the group is an any hit only when it is empty )
            std::string combinePath;        // the compute shader that merges two cascades
        };

        explicit RadianceCascade(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            if(!info.as || !info.manager || !info.probes || !info.lights){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RadianceCascade , since the acceleration structure , the texture manager , the probe manager or the light manager is nullptr"});
                THROW_MESSAGE;
            }

            if(info.placementPath.empty() || info.rayGenerationPath.empty() || info.missPath.empty() || info.shadowMissPath.empty() || info.hitPath.empty() || info.shadowAnyHitPath.empty() || info.combinePath.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RadianceCascade , since one of the required shader paths is empty ( placement , ray generation , miss , shadow miss , hit , shadow any hit , combine )"});
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~RadianceCascade() noexcept{
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs{};
            reqs.reserve(20);

            const auto image = [&reqs](const char* name , const ImageType type){
                GraphParameterInfo info{};
                info.type = moxGraphTypeImage;
                info.name = name;
                info.imageSpecification.type = type;
                reqs.push_back(info);
            };

            const auto buffer = [&reqs](const char* name){
                GraphParameterInfo info{};
                info.type = moxGraphTypeBuffer;
                info.name = name;
                reqs.push_back(info);
            };

            image("skybox" , ImageType::Skybox2);
            image("transmissionLUT" , ImageType::RenderImage2);
            image("multiScatter" , ImageType::RenderImage2);

            // the G buffer of the placement : the ping pong pairs , the one with the parity of the frame is the one of this frame
            image("positionImage_0" , ImageType::RenderImage2);
            image("positionImage_1" , ImageType::RenderImage2);
            image("normalImage_0" , ImageType::RenderImage2);
            image("normalImage_1" , ImageType::RenderImage2);
            image("pixelCheck" , ImageType::RenderImage2);

            buffer("generalBuffer");
            buffer("generalFeatures");
            buffer("forwardInfo");
            buffer("skyBuffer");
            buffer("sunBuffer");
            buffer("tileLightIndices");
            buffer("lightCountPerTile");
            buffer("lightBuffer");

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [RadianceCascade]"});

            const uint32_t frameIndex = inputData.context->sc->frameIndex;
            const uint8_t currentImage = frameIndex % 2;
            const uint32_t cascadeCount = static_cast<uint32_t>(cascadeInfos.size());

            // ------------------------------------------------------------ 1. the probes stand on the surfaces
            {
                VkShaderEXT shaders[1] = {placePipe->shaders[0]->shader};
                VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
                cmdBindShadersEXT(cmd , 1 , stages , shaders);

                const VkDescriptorBufferBindingInfoEXT info = descriptorBufferInfo(placePipe->buffers[currentImage]->pBuffer);
                cmdBindDescriptorBuffersEXT(cmd , 1 , &info);

                const uint32_t sets[1] = {0};
                const VkDeviceSize offsets[1] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , placePipe->layout->layout , 0 , 1 , sets , offsets);

                for(uint32_t cascade = 0 ; cascade < cascadeCount ; cascade++){
                    const RC::CascadeInfo& cs = cascadeInfos[cascade];
                    vkCmdPushConstants(cmd , placePipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(uint32_t) , &cascade);

                    // a thread for a probe : the probes on x , the probes on y
                    const uint32_t probesX = cs.CASCADE_X / cs.BUILD_SIZE;
                    const uint32_t probesY = cs.CASCADE_Y / cs.BUILD_SIZE;
                    vkCmdDispatch(cmd , (probesX + 7) / 8 , (probesY + 7) / 8 , 1);
                }

                shaders[0] = nullptr;
                cmdBindShadersEXT(cmd , 1 , stages , shaders);
            }

            // the rays start in the positions that the placement has just written
            memoryBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT , VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

            // ------------------------------------------------------------ 2. the rays of every cascade
            vkCmdBindPipeline(cmd , VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR , rtPipe->pipeline);
            {
                std::array<VkDescriptorBufferBindingInfoEXT , 4> info{};
                info[0] = descriptorBufferInfo(rtPipe->buffers[0]->pBuffer);
                info[1] = descriptorBufferInfo(inputData.manager->dbuffer->pBuffer);
                info[2] = descriptorBufferInfo(inputData.probes->buffer->pBuffer);
                info[3] = descriptorBufferInfo(inputData.lights->dbuffer->pBuffer);
                cmdBindDescriptorBuffersEXT(cmd , 4 , info.data());

                const uint32_t buffers[4] = {0 , 1 , 2 , 3};
                const VkDeviceSize offsets[4] = {0 , 0 , 0 , 0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR , rtPipe->layout->layout , 0 , 4 , buffers , offsets);
            }

            for(uint32_t cascade = 0 ; cascade < cascadeCount ; cascade++){
                const RC::CascadeInfo& info = cascadeInfos[cascade];
                vkCmdPushConstants(cmd , rtPipe->layout->layout , VK_SHADER_STAGE_RAYGEN_BIT_KHR , 0 , sizeof(uint32_t) , &cascade);
                // a ray for every texel of the image of the cascade . vkCmdTraceRaysKHR takes the grid of the rays
                sbt->traceRays(cmd , info.CASCADE_X , info.CASCADE_Y , 1);
            }

            // the rays wrote the images of the cascades , the merge reads and writes them
            memoryBarrier(cmd , VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            // ------------------------------------------------------------ 3. the merge from the top
            {
                VkShaderEXT shaders[1] = {combinePipe->shaders[0]->shader};
                VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
                cmdBindShadersEXT(cmd , 1 , stages , shaders);

                // the shader declares the set of the probes as the set 0 : its pipeline layout has only this set
                const VkDescriptorBufferBindingInfoEXT info = descriptorBufferInfo(inputData.probes->buffer->pBuffer);
                cmdBindDescriptorBuffersEXT(cmd , 1 , &info);

                const uint32_t buffers[1] = {0};
                const VkDeviceSize offsets[1] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , combinePipe->layout->layout , 0 , 1 , buffers , offsets);

                for(uint32_t current = cascadeCount - 1 ; current-- > 0 ;){
                    CombinePush push{};
                    push.current = current;
                    push.prev = current + 1;
                    // the cascade 0 is the image with the parity of the frame , the others have their own ( RCProbeManager::imageIndex )
                    push.currentImage = RCProbeManager::imageIndex(current , frameIndex);
                    push.prevImage = RCProbeManager::imageIndex(current + 1 , frameIndex);
                    vkCmdPushConstants(cmd , combinePipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(CombinePush) , &push);

                    const RC::CascadeInfo& cs = cascadeInfos[current];
                    vkCmdDispatch(cmd , (cs.CASCADE_X + 7) / 8 , (cs.CASCADE_Y + 7) / 8 , 1);

                    // the next pair reads the cascade that this one has just written
                    memoryBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                }

                shaders[0] = nullptr;
                cmdBindShadersEXT(cmd , 1 , stages , shaders);
            }

            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            return true;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindSources(sources ,
                GraphSlot<Skybox2>{"skybox" , &skybox} ,
                GraphSlot<RenderImage2>{"transmissionLUT" , &transmissionLUT} , GraphSlot<RenderImage2>{"multiScatter" , &multiScatter} ,
                GraphSlot<RenderImage2>{"positionImage_0" , &positionImage[0]} , GraphSlot<RenderImage2>{"positionImage_1" , &positionImage[1]} ,
                GraphSlot<RenderImage2>{"normalImage_0" , &normalImage[0]} , GraphSlot<RenderImage2>{"normalImage_1" , &normalImage[1]} ,
                GraphSlot<RenderImage2>{"pixelCheck" , &pixelCheck} ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer} , GraphSlot<BufferBase2>{"generalFeatures" , &features} ,
                GraphSlot<BufferBase2>{"forwardInfo" , &forwardInfo} , GraphSlot<BufferBase2>{"skyBuffer" , &skyBuffer} ,
                GraphSlot<BufferBase2>{"sunBuffer" , &sunBuffer} ,
                GraphSlot<BufferBase2>{"tileLightIndices" , &tileLightIndices} , GraphSlot<BufferBase2>{"lightCountPerTile" , &lightCountPerTile} ,
                GraphSlot<BufferBase2>{"lightBuffer" , &lightBuffer});
            if(!bound.has_value()) return bound;

            // the info of the cascades : the manager keeps it , the buffer is not read back
            {
                const RCProbeManager* probes = inputData.probes;
                if(probes->cascades.empty() || !probes->cascadeBuffer || !probes->buffer || !probes->layout){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create RadianceCascade , since the probe manager has no cascades or no descriptors"});
                }
                cascadeInfos.assign(probes->cascadesData.begin() , probes->cascadesData.end());
            }

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [RadianceCascade]"});

            const uint8_t id = inputData.context->sc->frameIndex % 2;

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 write = VK_ACCESS_2_SHADER_WRITE_BIT;
            constexpr VkAccessFlags2 uniformRead = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            // the number of the cascades is known only when the pass is made : the barriers are in vectors
            std::vector<VkImageMemoryBarrier2> imageBar{};
            imageBar.reserve(6 + inputData.probes->cascades.size());

            // sampled : the sky , the transmittance of the sun and the LUT of the multi scattering
            imageBar.push_back(skybox->translateImage(read , stage));
            imageBar.push_back(transmissionLUT->translateImage(read , stage));
            imageBar.push_back(multiScatter->translateImage(read , stage));
            // the G buffer of this frame : the placement reads it
            imageBar.push_back(positionImage[id]->translateImage(read , stage));
            imageBar.push_back(normalImage[id]->translateImage(read , stage));
            imageBar.push_back(pixelCheck->translateImage(read , stage));
            // the images of the cascades : the rays write them , the merge reads and writes them
            for(auto& cascade : inputData.probes->cascades){
                imageBar.push_back(cascade->translateImage(read | write , stage));
            }

            std::vector<VkBufferMemoryBarrier2> bufBar{};
            bufBar.reserve(20);

            bufBar.push_back(generalBuffer->translateBuffer(uniformRead , stage));
            bufBar.push_back(features->translateBuffer(uniformRead , stage));
            bufBar.push_back(forwardInfo->translateBuffer(uniformRead , stage));
            bufBar.push_back(skyBuffer->translateBuffer(uniformRead , stage));
            bufBar.push_back(sunBuffer->translateBuffer(uniformRead , stage));

            // the lamps of the tiles ( the bindings 6 7 and 8 of the set 0 are still there ) and the world grid of the lamps ( the set 3 )
            bufBar.push_back(tileLightIndices->translateBuffer(read , stage));
            bufBar.push_back(lightCountPerTile->translateBuffer(read , stage));
            bufBar.push_back(lightBuffer->translateBuffer(read , stage));
            bufBar.push_back(inputData.lights->lightCellBuffer->translateBuffer(read , stage));
            bufBar.push_back(inputData.lights->lightCascadeBuffer->translateBuffer(uniformRead , stage));

            // the geometry and the materials : the closest hit reads them
            bufBar.push_back(Mesh2::vertexBuffer->translateBuffer(read , stage));
            bufBar.push_back(Mesh2::indicesBuffer->translateBuffer(read , stage));
            bufBar.push_back(Mesh2::instanceLayoutBuffer->translateBuffer(read , stage));
            bufBar.push_back(Mesh2::instanceBuffer->translateBuffer(read , stage));
            bufBar.push_back(Mesh2::MeshBuffer->translateBuffer(read , stage));
            bufBar.push_back(Mesh2::LUT_buffer->translateBuffer(read , stage));

            // the buffers of the probes : the cascades and the general data are written once , the probes are written by the placement and read by everything after it
            bufBar.push_back(inputData.probes->probesBuffer->translateBuffer(read | write , stage));
            bufBar.push_back(inputData.probes->cascadeBuffer->translateBuffer(read , stage));
            bufBar.push_back(inputData.probes->generalProbeBuffer->translateBuffer(uniformRead , stage));

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.pNext = nullptr;
            dep.dependencyFlags = 0;
            dep.memoryBarrierCount = 0;
            dep.pMemoryBarriers = nullptr;
            dep.bufferMemoryBarrierCount = static_cast<uint32_t>(bufBar.size());
            dep.pBufferMemoryBarriers = bufBar.data();
            dep.imageMemoryBarrierCount = static_cast<uint32_t>(imageBar.size());
            dep.pImageMemoryBarriers = imageBar.data();

            vkCmdPipelineBarrier2(cmd , &dep);

            return {};
        }

    private:

        // the push constant of CombineCascades.hlsl : the cascades and their images ( the cascade 0 has two images , see RCProbeManager )
        struct CombinePush{
            uint32_t current;
            uint32_t prev;
            uint32_t currentImage;
            uint32_t prevImage;
        };

        pPipeline rtPipe{nullptr};
        pSBT2 sbt{nullptr};
        pSampler sampler{nullptr};

        // the compute shader of the merge : its layout is the set of the probes only ( the shader declares it as the set 0 )
        pShaderPipeType combinePipe{nullptr};
        // the compute shader of the placement : the set of its own , two descriptor buffers ( the parity of the frame : the G buffer is ping pong )
        pShaderPipeType placePipe{nullptr};

        CreateInfo inputData{};

        // the info of the cascades : a copy of the data of the probe manager
        std::vector<RC::CascadeInfo> cascadeInfos{};

        Skybox2* skybox{nullptr};
        RenderImage2* transmissionLUT{nullptr};
        RenderImage2* multiScatter{nullptr};

        RenderImage2* positionImage[2]{nullptr , nullptr};
        RenderImage2* normalImage[2]{nullptr , nullptr};
        RenderImage2* pixelCheck{nullptr};

        BufferBase2* generalBuffer{nullptr};
        BufferBase2* features{nullptr};
        BufferBase2* forwardInfo{nullptr};
        BufferBase2* skyBuffer{nullptr};
        BufferBase2* sunBuffer{nullptr};
        BufferBase2* tileLightIndices{nullptr};
        BufferBase2* lightCountPerTile{nullptr};
        BufferBase2* lightBuffer{nullptr};

        static VkDescriptorBufferBindingInfoEXT descriptorBufferInfo(const VkDeviceAddress address) noexcept{
            VkDescriptorBufferBindingInfoEXT info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
            info.pNext = nullptr;
            info.address = address;
            info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
            return info;
        }

        // everything that was written by the first stage is visible to the second one ( the images of the cascades are in the general layout )
        static void memoryBarrier(VkCommandBuffer& cmd , const VkPipelineStageFlags2 first , const VkPipelineStageFlags2 second) noexcept{
            VkMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            barrier.pNext = nullptr;
            barrier.srcStageMask = first;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask = second;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.pNext = nullptr;
            dep.dependencyFlags = 0;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &barrier;
            dep.bufferMemoryBarrierCount = 0;
            dep.pBufferMemoryBarriers = nullptr;
            dep.imageMemoryBarrierCount = 0;
            dep.pImageMemoryBarriers = nullptr;

            vkCmdPipelineBarrier2(cmd , &dep);
        }

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(RadianceCascade&& input) noexcept{
            if(this == &input) return;
            processDelete();

            rtPipe = std::move(input.rtPipe);
            sbt = std::move(input.sbt);
            sampler = std::move(input.sampler);
            combinePipe = std::move(input.combinePipe);
            placePipe = std::move(input.placePipe);
            // the acceleration structure , the texture manager , the probe manager and the light manager are not owned by the pass
            inputData = input.inputData;
            cascadeInfos = std::move(input.cascadeInfos);

            skybox = std::exchange(input.skybox , nullptr);
            transmissionLUT = std::exchange(input.transmissionLUT , nullptr);
            multiScatter = std::exchange(input.multiScatter , nullptr);

            for(const auto i : {0 , 1}){
                positionImage[i] = std::exchange(input.positionImage[i] , nullptr);
                normalImage[i] = std::exchange(input.normalImage[i] , nullptr);
            }
            pixelCheck = std::exchange(input.pixelCheck , nullptr);

            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            features = std::exchange(input.features , nullptr);
            forwardInfo = std::exchange(input.forwardInfo , nullptr);
            skyBuffer = std::exchange(input.skyBuffer , nullptr);
            sunBuffer = std::exchange(input.sunBuffer , nullptr);
            tileLightIndices = std::exchange(input.tileLightIndices , nullptr);
            lightCountPerTile = std::exchange(input.lightCountPerTile , nullptr);
            lightBuffer = std::exchange(input.lightBuffer , nullptr);

            // the moved from pass has no pipes any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            // the images and the buffers belong to the graph , they are only forgotten . the SBT is destroyed before the pipeline it was made from
            sbt = nullptr;
            rtPipe = nullptr;
            combinePipe = nullptr;
            placePipe = nullptr;
            sampler = nullptr;
            cascadeInfos.clear();

            skybox = nullptr;
            transmissionLUT = nullptr;
            multiScatter = nullptr;

            for(const auto i : {0 , 1}){
                positionImage[i] = nullptr;
                normalImage[i] = nullptr;
            }
            pixelCheck = nullptr;

            generalBuffer = nullptr;
            features = nullptr;
            forwardInfo = nullptr;
            skyBuffer = nullptr;
            sunBuffer = nullptr;
            tileLightIndices = nullptr;
            lightCountPerTile = nullptr;
            lightBuffer = nullptr;

            inputData = {};
            completedInit = false;
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            VulkanContext* context = inputData.context;

            try{
                Pipeline::CreateInfo pipeInfo{};
                pipeInfo.context = context;
                pipeInfo.cacheFile = "";
                rtPipe = std::make_shared<Pipeline>(pipeInfo);

                {
                    Sampler::CreateInfo info{};
                    info.context = context;
                    info.linearClampSampler = true;

                    sampler = std::make_shared<Sampler>(info);
                }

                // ------------------------------------------------------------ set 0 : the scene ( the numbers of the bindings are the ones of the shaders )
                {
                    std::vector<DescriptorLayoutData> data(19);

                    data[0] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR , 1};

                    for(uint32_t i = 1 ; i <= 5 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    for(uint32_t i = 6 ; i <= 14 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    data[15] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[16] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[17] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[18] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

                    rtPipe->addDescriptorLayout(info);
                }

                // ------------------------------------------------------------ the pipeline layout of the rays : set 0 the scene , set 1 the textures , set 2 the probes ,
                // set 3 the world grid of the lamps ; the cascade of the ray is the push constant of the ray generation shader
                {
                    VkPushConstantRange range{};
                    range.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
                    range.offset = 0;
                    range.size = sizeof(uint32_t);

                    PipelineLayout::CreateInfo info{};
                    info.context = context;
                    info.layouts = {rtPipe->descriptorLayouts[0].get() , inputData.manager->layout.get() , inputData.probes->layout.get() , inputData.lights->layout.get()};
                    info.ranges = {range};

                    rtPipe->layout = std::make_shared<PipelineLayout>(info);
                }

                // ------------------------------------------------------------ set 0 : the descriptor buffer
                {
                    std::vector<DescriptorData2> data(19);

                    data[0] = DescriptorData2::createData(inputData.as , 0 , 0);

                    data[1] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0);
                    data[2] = DescriptorData2::createData(forwardInfo , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0);
                    data[3] = DescriptorData2::createData(skyBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 3 , 0);
                    data[4] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 4 , 0);
                    data[5] = DescriptorData2::createData(sunBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 5 , 0);

                    data[6] = DescriptorData2::createData(tileLightIndices , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 6 , 0);
                    data[7] = DescriptorData2::createData(lightCountPerTile , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 7 , 0);
                    data[8] = DescriptorData2::createData(lightBuffer , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 8 , 0);

                    data[9] = DescriptorData2::createData(Mesh2::vertexBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 9 , 0);
                    data[10] = DescriptorData2::createData(Mesh2::indicesBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 10 , 0);
                    data[11] = DescriptorData2::createData(Mesh2::instanceLayoutBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 11 , 0);
                    data[12] = DescriptorData2::createData(Mesh2::instanceBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 12 , 0);
                    data[13] = DescriptorData2::createData(Mesh2::MeshBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 13 , 0);
                    data[14] = DescriptorData2::createData(Mesh2::LUT_buffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 14 , 0);

                    data[15] = DescriptorData2::createData(skybox , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 15 , 0);
                    data[16] = DescriptorData2::createData(sampler.get() , 16 , 0);
                    data[17] = DescriptorData2::createData(transmissionLUT , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 17 , 0);
                    data[18] = DescriptorData2::createData(multiScatter , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 18 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = rtPipe->descriptorLayouts[0].get();

                    rtPipe->addDescriptorBuffer2(info);
                }

                // ------------------------------------------------------------ the ray tracing shaders
                ShaderModuleType rayGeneration(context , inputData.rayGenerationPath);
                ShaderModuleType miss(context , inputData.missPath);
                ShaderModuleType shadowMiss(context , inputData.shadowMissPath);
                ShaderModuleType hit(context , inputData.hitPath);
                ShaderModuleType shadowAnyHit(context , inputData.shadowAnyHitPath);
                std::optional<ShaderModuleType> shadowHit;
                if(!inputData.shadowHitPath.empty()) shadowHit.emplace(context , inputData.shadowHitPath);

                // the SBT stores no names , only the order : THIS ORDER is what the miss index and the hit offset of TraceRay mean
                std::vector<VkPipelineShaderStageCreateInfo> stages{};
                stages.reserve(6);

                const auto addStage = [&stages](const VkShaderStageFlagBits stage , const VkShaderModule module) -> uint32_t{
                    VkPipelineShaderStageCreateInfo info{};
                    info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    info.pNext = nullptr;
                    info.flags = 0;
                    info.stage = stage;
                    info.module = module;
                    info.pName = "main";
                    info.pSpecializationInfo = nullptr;
                    stages.push_back(info);
                    return static_cast<uint32_t>(stages.size() - 1);
                };

                const uint32_t rayGenerationStage = addStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR , rayGeneration.module);
                const uint32_t missStage = addStage(VK_SHADER_STAGE_MISS_BIT_KHR , miss.module);
                const uint32_t shadowMissStage = addStage(VK_SHADER_STAGE_MISS_BIT_KHR , shadowMiss.module);
                const uint32_t hitStage = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR , hit.module);
                const uint32_t shadowAnyHitStage = addStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR , shadowAnyHit.module);
                const uint32_t shadowHitStage = shadowHit.has_value() ? addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR , shadowHit->module) : VK_SHADER_UNUSED_KHR;

                const auto generalGroup = [](const uint32_t stage){
                    VkRayTracingShaderGroupCreateInfoKHR group{};
                    group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                    group.pNext = nullptr;
                    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
                    group.generalShader = stage;
                    group.closestHitShader = VK_SHADER_UNUSED_KHR;
                    group.anyHitShader = VK_SHADER_UNUSED_KHR;
                    group.intersectionShader = VK_SHADER_UNUSED_KHR;
                    group.pShaderGroupCaptureReplayHandle = nullptr;
                    return group;
                };

                const auto hitGroup = [](const uint32_t closestHit , const uint32_t anyHit){
                    VkRayTracingShaderGroupCreateInfoKHR group{};
                    group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                    group.pNext = nullptr;
                    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                    group.generalShader = VK_SHADER_UNUSED_KHR;
                    group.closestHitShader = closestHit;
                    group.anyHitShader = anyHit;
                    group.intersectionShader = VK_SHADER_UNUSED_KHR;
                    group.pShaderGroupCaptureReplayHandle = nullptr;
                    return group;
                };

                const std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups = {
                    generalGroup(rayGenerationStage) ,
                    generalGroup(missStage) ,                            // miss 0 : the sky
                    generalGroup(shadowMissStage) ,                      // miss 1 : the shadow
                    hitGroup(hitStage , VK_SHADER_UNUSED_KHR) ,          // hit 0 : the closest hit
                    hitGroup(shadowHitStage , shadowAnyHitStage)         // hit 1 : the shadow
                };

                VkRayTracingPipelineCreateInfoKHR info{};
                info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
                info.pNext = nullptr;
                // the descriptors of this pipeline are descriptor buffers
                info.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                info.stageCount = static_cast<uint32_t>(stages.size());
                info.pStages = stages.data();
                info.groupCount = static_cast<uint32_t>(groups.size());
                info.pGroups = groups.data();
                // the ray generation shader traces the ray ( depth 1 ) and the closest hit traces the shadow ray ( depth 2 )
                info.maxPipelineRayRecursionDepth = 2;
                info.pLibraryInfo = nullptr;
                info.pLibraryInterface = nullptr;
                info.pDynamicState = nullptr;
                info.layout = rtPipe->layout->layout;
                info.basePipelineHandle = nullptr;
                info.basePipelineIndex = -1;

                const auto result = createRayTracingPipelinesKHR(context->logicalDevice->device , nullptr , nullptr , 1 , &info , nullptr , &rtPipe->pipeline);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the ray tracing pipeline [RadianceCascade] , error ID -> {}" , static_cast<int32_t>(result))});
                }

                // ------------------------------------------------------------ the shader binding table : the order of the groups is the one above
                {
                    SBT2::CreateInfo sbtInfo{};
                    sbtInfo.context = context;
                    sbtInfo.pipeline = rtPipe->pipeline;
                    sbtInfo.groupTypes = {
                        SBTGroupType::RayGeneration ,
                        SBTGroupType::Miss , SBTGroupType::Miss ,
                        SBTGroupType::Hit , SBTGroupType::Hit
                    };

                    sbt = std::make_shared<SBT2>(sbtInfo);
                }

                // ------------------------------------------------------------ the merge : a compute shader , the set of the probes is its set 0
                {
                    combinePipe = std::make_shared<ShaderPipeType>(context);

                    VkPushConstantRange range{};
                    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                    range.offset = 0;
                    range.size = sizeof(CombinePush);

                    PipelineLayout::CreateInfo layoutInfo{};
                    layoutInfo.context = context;
                    layoutInfo.layouts = {inputData.probes->layout.get()};
                    layoutInfo.ranges = {range};

                    combinePipe->layout = std::make_shared<PipelineLayout>(layoutInfo);

                    ShaderObjectType::CreateInfo shaderInfo{};
                    shaderInfo.context = context;
                    shaderInfo.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    shaderInfo.nextStage = 0;
                    shaderInfo.filePath = inputData.combinePath;
                    shaderInfo.layouts = {inputData.probes->layout.get()};
                    shaderInfo.pushes = {range};

                    combinePipe->addShader(shaderInfo);
                }

                // ------------------------------------------------------------ the placement : a compute shader with a set of its own ( RCplacement.hlsl )
                {
                    placePipe = std::make_shared<ShaderPipeType>(context);

                    {
                        std::vector<DescriptorLayoutData> data(6);
                        data[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                        data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                        data[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                        data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                        data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                        data[5] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                        DescriptorLayout::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                        info.pNext = nullptr;
                        info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                        placePipe->addDescriptorLayout(info);
                    }

                    VkPushConstantRange range{};
                    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                    range.offset = 0;
                    range.size = sizeof(uint32_t);

                    {
                        PipelineLayout::CreateInfo layoutInfo{};
                        layoutInfo.context = context;
                        layoutInfo.layouts = {placePipe->descriptorLayouts[0].get()};
                        layoutInfo.ranges = {range};

                        placePipe->layout = std::make_shared<PipelineLayout>(layoutInfo);
                    }

                    // the G buffer is ping pong : the descriptor buffer with the parity of the frame holds the images of this frame
                    for(const auto id : {0 , 1}){
                        std::vector<DescriptorData2> data(6);
                        data[0] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                        data[1] = DescriptorData2::createData(normalImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0);
                        data[2] = DescriptorData2::createData(positionImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0);
                        data[3] = DescriptorData2::createData(pixelCheck , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);
                        data[4] = DescriptorData2::createData(inputData.probes->probesBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0);
                        data[5] = DescriptorData2::createData(inputData.probes->cascadeBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 5 , 0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = placePipe->descriptorLayouts[0].get();

                        placePipe->addDescriptorBuffer2(info);
                    }

                    ShaderObjectType::CreateInfo shaderInfo{};
                    shaderInfo.context = context;
                    shaderInfo.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    shaderInfo.nextStage = 0;
                    shaderInfo.filePath = inputData.placementPath;
                    shaderInfo.layouts = {placePipe->descriptorLayouts[0].get()};
                    shaderInfo.pushes = {range};

                    placePipe->addShader(shaderInfo);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RadianceCascade pipe , please check the logs above"});
            }

            completedInit = true;
            return {};
        }
    };

    using pRadianceCascade = std::shared_ptr<RadianceCascade>;
}
