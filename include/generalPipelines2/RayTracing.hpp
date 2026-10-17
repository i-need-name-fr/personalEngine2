#pragma once

#include "../Pipeline/Pipeline.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../AS2/AccelerationStructureBase.hpp"
#include "../Object2/Mesh2.hpp"
#include "../Buffer2/SBT2.hpp"
#include "../ShaderModule/ShaderModule.hpp"
#include "../Sampler/Sampler.hpp"
#include "../Image2/TextureManager2.hpp"

namespace mox{

    // =============================== INFO ===============================
    // RayTracing - the ray tracing pass that runs AFTER the Deferred render : every pixel comes from the G buffer , there are no primary rays
    // ( the ray generation shader reads the G buffer , makes a GeneralObjectInfo and calls generateReservoirGI )
    //
    // the pipeline is a REAL ray tracing pipeline ( vkCreateRayTracingPipelinesKHR ) , its descriptors are descriptor buffers :
    //   set 0 : 37 bindings , the same list as in every shader of Shaders/RayTracing ( see the listing in the shaders ) , two descriptor buffers , the
    //           one with the index frameIndex % 2 is bound ( the ping pong images and the reservoirs are swapped inside )
    //   set 1 : the textures of the TextureManager2 ( the same set as the Deferred render )
    //
    // the order of the groups = the numbers in shaderStructure.h ( RT_MISS_* and RT_HITGROUP_* ) :
    //   miss : 0 general , 1 hit info ( GI ) , 2 shadow
    //   hit  : 0 general , 1 hit info ( GI ) , 2 shadow ( an any hit only )
    // the GENERAL ones are optional : with no path the general miss is the GI miss and the general hit group is an empty one ( no shader )
    //
    // what the barriers of the pass say : the G buffer , the sky and the buffers of the scene are read , the images of the light ( diffuse ,
    // specular , indirect ) are read and written , the reservoirs of GI : this frame writes its own and reads the ones of the previous frame
    // =============================== END ===============================

    class RayTracing : public GraphPass2{
    public:

        SetTypicalOperators(RayTracing)
        SetTypicalTransferOperators(RayTracing)

        struct CreateInfo{
            VulkanContext* context{};
            AccelerationStructure* as{};
            TextureManager2* manager{};

            std::string rayGenerationPath;

            // miss shaders
            std::string giMissPath;          // RT_MISS_HITINFO
            std::string shadowMissPath;      // RT_MISS_SHADOW
            std::string generalMissPath;     // RT_MISS_GENERAL , optional : the GI miss is used when it is empty

            // hit shaders
            std::string giHitPath;           // RT_HITGROUP_HITINFO : the closest hit
            std::string shadowAnyHitPath;    // RT_HITGROUP_SHADOW : the any hit
            std::string generalHitPath;      // RT_HITGROUP_GENERAL , optional : an empty hit group when it is empty
        };

        explicit RayTracing(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            if(!info.as){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RayTracing , since the acceleration structure is nullptr"});
                THROW_MESSAGE;
            }

            if(!info.manager){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RayTracing , since the texture manager is nullptr"});
                THROW_MESSAGE;
            }

            if(info.rayGenerationPath.empty() || info.giMissPath.empty() || info.shadowMissPath.empty() || info.giHitPath.empty() || info.shadowAnyHitPath.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RayTracing , since one of the required shader paths is empty ( ray generation , GI miss , shadow miss , GI hit , shadow any hit )"});
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~RayTracing() noexcept{
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs{};
            reqs.reserve(48);

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

            // the ping pong pairs
            image("positionImage_0" , ImageType::RenderImage2);
            image("positionImage_1" , ImageType::RenderImage2);
            image("normalImage_0" , ImageType::RenderImage2);
            image("normalImage_1" , ImageType::RenderImage2);
            image("depthImage_0" , ImageType::RenderImage2);
            image("depthImage_1" , ImageType::RenderImage2);
            image("diffuseImage_0" , ImageType::RenderImage2);
            image("diffuseImage_1" , ImageType::RenderImage2);
            image("specularImage_0" , ImageType::RenderImage2);
            image("specularImage_1" , ImageType::RenderImage2);
            image("indirectImage_0" , ImageType::RenderImage2);
            image("indirectImage_1" , ImageType::RenderImage2);
            image("shadowImage_0" , ImageType::RenderImage2);
            image("shadowImage_1" , ImageType::RenderImage2);

            image("albedoImage" , ImageType::RenderImage2);
            image("roughnessImage" , ImageType::RenderImage2);
            image("metallicImage" , ImageType::RenderImage2);
            image("viewImage" , ImageType::RenderImage2);
            image("motionVector" , ImageType::RenderImage2);
            image("pixelCheck" , ImageType::RenderImage2);

            image("skybox" , ImageType::Skybox2);
            image("transmissionLUT" , ImageType::RenderImage2);
            image("multiScatter" , ImageType::RenderImage2);

            buffer("generalBuffer");
            buffer("generalFeatures");
            buffer("forwardInfo");
            buffer("skyBuffer");
            buffer("sunBuffer");
            buffer("tileLightIndices");
            buffer("lightCountPerTile");
            buffer("lightBuffer");
            buffer("ReSTIRDIBuffer_0");
            buffer("ReSTIRDIBuffer_1");
            buffer("ReSTIRGIBuffer_0");
            buffer("ReSTIRGIBuffer_1");

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [RayTracing]"});

            const uint8_t currentImage = inputData.context->sc->frameIndex % 2;

            vkCmdBindPipeline(cmd , VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR , pipe->pipeline);

            // set 0 : the descriptor buffer of this pass , set 1 : the textures
            {
                std::array<VkDescriptorBufferBindingInfoEXT , 2> info{};

                info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[0].address = pipe->buffers[currentImage]->pBuffer;
                info[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[0].pNext = nullptr;

                info[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[1].address = inputData.manager->dbuffer->pBuffer;
                info[1].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[1].pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 2 , info.data());

                const uint32_t sets[2] = {0 , 1};
                const VkDeviceSize offsets[2] = {0 , 0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR , pipe->layout->layout , 0 , 2 , sets , offsets);
            }

            // vkCmdTraceRaysKHR takes the GRID of the rays , not the number of the workgroups : there is nothing to divide by 8
            sbt->traceRays(cmd , inputData.context->sc->extent.width , inputData.context->sc->extent.height , 1);

            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            return true;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindSources(sources ,
                GraphSlot<RenderImage2>{"positionImage_0" , &positionImage[0]} , GraphSlot<RenderImage2>{"positionImage_1" , &positionImage[1]} ,
                GraphSlot<RenderImage2>{"normalImage_0" , &normalImage[0]} , GraphSlot<RenderImage2>{"normalImage_1" , &normalImage[1]} ,
                GraphSlot<RenderImage2>{"depthImage_0" , &depthImage[0]} , GraphSlot<RenderImage2>{"depthImage_1" , &depthImage[1]} ,
                GraphSlot<RenderImage2>{"diffuseImage_0" , &diffuseImage[0]} , GraphSlot<RenderImage2>{"diffuseImage_1" , &diffuseImage[1]} ,
                GraphSlot<RenderImage2>{"specularImage_0" , &specularImage[0]} , GraphSlot<RenderImage2>{"specularImage_1" , &specularImage[1]} ,
                GraphSlot<RenderImage2>{"indirectImage_0" , &indirectImage[0]} , GraphSlot<RenderImage2>{"indirectImage_1" , &indirectImage[1]} ,
                GraphSlot<RenderImage2>{"shadowImage_0" , &shadowImage[0]} , GraphSlot<RenderImage2>{"shadowImage_1" , &shadowImage[1]} ,
                GraphSlot<RenderImage2>{"albedoImage" , &albedoImage} , GraphSlot<RenderImage2>{"roughnessImage" , &roughnessImage} ,
                GraphSlot<RenderImage2>{"metallicImage" , &metallicImage} , GraphSlot<RenderImage2>{"viewImage" , &viewImage} ,
                GraphSlot<RenderImage2>{"motionVector" , &motionVector} , GraphSlot<RenderImage2>{"pixelCheck" , &pixelCheck} ,
                GraphSlot<Skybox2>{"skybox" , &skybox} ,
                GraphSlot<RenderImage2>{"transmissionLUT" , &transmissionLUT} , GraphSlot<RenderImage2>{"multiScatter" , &multiScatter} ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer} , GraphSlot<BufferBase2>{"generalFeatures" , &features} ,
                GraphSlot<BufferBase2>{"forwardInfo" , &forwardInfo} , GraphSlot<BufferBase2>{"skyBuffer" , &skyBuffer} ,
                GraphSlot<BufferBase2>{"sunBuffer" , &sunBuffer} ,
                GraphSlot<BufferBase2>{"tileLightIndices" , &tileLightIndices} , GraphSlot<BufferBase2>{"lightCountPerTile" , &lightCountPerTile} ,
                GraphSlot<BufferBase2>{"lightBuffer" , &lightBuffer} ,
                GraphSlot<BufferBase2>{"ReSTIRDIBuffer_0" , &ReSTIRDIBuffer[0]} , GraphSlot<BufferBase2>{"ReSTIRDIBuffer_1" , &ReSTIRDIBuffer[1]} ,
                GraphSlot<BufferBase2>{"ReSTIRGIBuffer_0" , &ReSTIRGIBuffer[0]} , GraphSlot<BufferBase2>{"ReSTIRGIBuffer_1" , &ReSTIRGIBuffer[1]});
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [RayTracing]"});

            const uint8_t id = inputData.context->sc->frameIndex % 2;
            const uint8_t prev = 1 - id;

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 write = VK_ACCESS_2_SHADER_WRITE_BIT;
            constexpr VkAccessFlags2 uniformRead = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            // the G buffer of this frame and the sky : only read . the images of the light : the shaders add to them , so read and write
            // ( the images that the shaders only bind , like the shadows and the view image , are not touched by this pass )
            std::array<VkImageMemoryBarrier2 , 14> imageBar;

            imageBar[0] = positionImage[id]->translateImage(read , stage);
            imageBar[1] = normalImage[id]->translateImage(read , stage);
            imageBar[2] = depthImage[id]->translateImage(read , stage);
            imageBar[3] = albedoImage->translateImage(read , stage);
            imageBar[4] = roughnessImage->translateImage(read , stage);
            imageBar[5] = metallicImage->translateImage(read , stage);
            imageBar[6] = motionVector->translateImage(read , stage);
            imageBar[7] = pixelCheck->translateImage(read , stage);

            // sampled : the sky , the transmittance of the sun and the LUT of the multi scattering
            imageBar[8] = skybox->translateImage(read , stage);
            imageBar[9] = transmissionLUT->translateImage(read , stage);
            imageBar[10] = multiScatter->translateImage(read , stage);

            imageBar[11] = diffuseImage[id]->translateImage(read | write , stage);
            imageBar[12] = specularImage[id]->translateImage(read | write , stage);
            imageBar[13] = indirectImage[id]->translateImage(read | write , stage);

            std::array<VkBufferMemoryBarrier2 , 16> bufBar;

            bufBar[0] = generalBuffer->translateBuffer(uniformRead , stage);
            bufBar[1] = features->translateBuffer(uniformRead , stage);
            bufBar[2] = forwardInfo->translateBuffer(uniformRead , stage);
            bufBar[3] = skyBuffer->translateBuffer(uniformRead , stage);
            bufBar[4] = sunBuffer->translateBuffer(uniformRead , stage);

            // the lamps of the tiles : written by the ForwardPlus
            bufBar[5] = tileLightIndices->translateBuffer(read , stage);
            bufBar[6] = lightCountPerTile->translateBuffer(read , stage);
            bufBar[7] = lightBuffer->translateBuffer(read , stage);

            // the geometry and the materials : the closest hit reads them
            bufBar[8] = Mesh2::vertexBuffer->translateBuffer(read , stage);
            bufBar[9] = Mesh2::indicesBuffer->translateBuffer(read , stage);
            bufBar[10] = Mesh2::instanceLayoutBuffer->translateBuffer(read , stage);
            bufBar[11] = Mesh2::instanceBuffer->translateBuffer(read , stage);
            bufBar[12] = Mesh2::MeshBuffer->translateBuffer(read , stage);
            bufBar[13] = Mesh2::LUT_buffer->translateBuffer(read , stage);

            // the reservoirs of GI : this frame writes its own ( and may read it , the spatial reuse of the same pass ) and reads the previous one
            bufBar[14] = ReSTIRGIBuffer[id]->translateBuffer(read | write , stage);
            bufBar[15] = ReSTIRGIBuffer[prev]->translateBuffer(read , stage);

            auto dep = TypicalDependency<16 , 14>(imageBar.data() , bufBar.data());

            vkCmdPipelineBarrier2(cmd , &dep);

            return {};
        }

    private:

        pPipeline pipe{nullptr};
        pSBT2 sbt{nullptr};
        pSampler sampler{nullptr};

        CreateInfo inputData{};

        RenderImage2* positionImage[2]{nullptr , nullptr};
        RenderImage2* normalImage[2]{nullptr , nullptr};
        RenderImage2* depthImage[2]{nullptr , nullptr};
        RenderImage2* diffuseImage[2]{nullptr , nullptr};
        RenderImage2* specularImage[2]{nullptr , nullptr};
        RenderImage2* indirectImage[2]{nullptr , nullptr};
        RenderImage2* shadowImage[2]{nullptr , nullptr};

        RenderImage2* albedoImage{nullptr};
        RenderImage2* roughnessImage{nullptr};
        RenderImage2* metallicImage{nullptr};
        RenderImage2* viewImage{nullptr};
        RenderImage2* motionVector{nullptr};
        RenderImage2* pixelCheck{nullptr};

        Skybox2* skybox{nullptr};
        RenderImage2* transmissionLUT{nullptr};
        RenderImage2* multiScatter{nullptr};

        BufferBase2* generalBuffer{nullptr};
        BufferBase2* features{nullptr};
        BufferBase2* forwardInfo{nullptr};
        BufferBase2* skyBuffer{nullptr};
        BufferBase2* sunBuffer{nullptr};
        BufferBase2* tileLightIndices{nullptr};
        BufferBase2* lightCountPerTile{nullptr};
        BufferBase2* lightBuffer{nullptr};
        BufferBase2* ReSTIRDIBuffer[2]{nullptr , nullptr};
        BufferBase2* ReSTIRGIBuffer[2]{nullptr , nullptr};

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(RayTracing&& input) noexcept{
            if(this == &input) return;
            processDelete();

            pipe = std::move(input.pipe);
            sbt = std::move(input.sbt);
            sampler = std::move(input.sampler);
            // the acceleration structure and the texture manager are not owned by the pass
            inputData = input.inputData;

            for(const auto i : {0 , 1}){
                positionImage[i] = std::exchange(input.positionImage[i] , nullptr);
                normalImage[i] = std::exchange(input.normalImage[i] , nullptr);
                depthImage[i] = std::exchange(input.depthImage[i] , nullptr);
                diffuseImage[i] = std::exchange(input.diffuseImage[i] , nullptr);
                specularImage[i] = std::exchange(input.specularImage[i] , nullptr);
                indirectImage[i] = std::exchange(input.indirectImage[i] , nullptr);
                shadowImage[i] = std::exchange(input.shadowImage[i] , nullptr);
                ReSTIRDIBuffer[i] = std::exchange(input.ReSTIRDIBuffer[i] , nullptr);
                ReSTIRGIBuffer[i] = std::exchange(input.ReSTIRGIBuffer[i] , nullptr);
            }

            albedoImage = std::exchange(input.albedoImage , nullptr);
            roughnessImage = std::exchange(input.roughnessImage , nullptr);
            metallicImage = std::exchange(input.metallicImage , nullptr);
            viewImage = std::exchange(input.viewImage , nullptr);
            motionVector = std::exchange(input.motionVector , nullptr);
            pixelCheck = std::exchange(input.pixelCheck , nullptr);

            skybox = std::exchange(input.skybox , nullptr);
            transmissionLUT = std::exchange(input.transmissionLUT , nullptr);
            multiScatter = std::exchange(input.multiScatter , nullptr);

            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            features = std::exchange(input.features , nullptr);
            forwardInfo = std::exchange(input.forwardInfo , nullptr);
            skyBuffer = std::exchange(input.skyBuffer , nullptr);
            sunBuffer = std::exchange(input.sunBuffer , nullptr);
            tileLightIndices = std::exchange(input.tileLightIndices , nullptr);
            lightCountPerTile = std::exchange(input.lightCountPerTile , nullptr);
            lightBuffer = std::exchange(input.lightBuffer , nullptr);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            // the images and the buffers belong to the graph , they are only forgotten . the SBT is destroyed before the pipeline it was made from
            sbt = nullptr;
            pipe = nullptr;
            sampler = nullptr;

            for(const auto i : {0 , 1}){
                positionImage[i] = nullptr;
                normalImage[i] = nullptr;
                depthImage[i] = nullptr;
                diffuseImage[i] = nullptr;
                specularImage[i] = nullptr;
                indirectImage[i] = nullptr;
                shadowImage[i] = nullptr;
                ReSTIRDIBuffer[i] = nullptr;
                ReSTIRGIBuffer[i] = nullptr;
            }

            albedoImage = nullptr;
            roughnessImage = nullptr;
            metallicImage = nullptr;
            viewImage = nullptr;
            motionVector = nullptr;
            pixelCheck = nullptr;

            skybox = nullptr;
            transmissionLUT = nullptr;
            multiScatter = nullptr;

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
                pipe = std::make_shared<Pipeline>(pipeInfo);

                {
                    Sampler::CreateInfo info{};
                    info.context = context;
                    info.linearClampSampler = true;

                    sampler = std::make_shared<Sampler>(info);
                }

                {
                    std::vector<DescriptorLayoutData> data(37);

                    data[0] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR , 1};

                    for(uint32_t i = 1 ; i <= 5 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    for(uint32_t i = 6 ; i <= 18 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    data[19] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[20] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[21] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[22] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    for(uint32_t i = 23 ; i <= 36 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

                    pipe->addDescriptorLayout(info);
                }

                // ------------------------------------------------------------ the pipeline layout : set 0 and the textures
                {
                    PipelineLayout::CreateInfo info{};
                    info.context = context;
                    info.layouts = {pipe->descriptorLayouts[0].get() , inputData.manager->layout.get()};
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                // ------------------------------------------------------------ set 0 : the descriptor buffers , one per parity of the frame
                for(const auto id : {0 , 1}){
                    std::vector<DescriptorData2> data(37);

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

                    // the reservoirs : this frame writes the one with its own parity and reads the other
                    data[15] = DescriptorData2::createData(ReSTIRDIBuffer[id] , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 15 , 0);
                    data[16] = DescriptorData2::createData(ReSTIRDIBuffer[!id] , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 16 , 0);
                    data[17] = DescriptorData2::createData(ReSTIRGIBuffer[id] , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 17 , 0);
                    data[18] = DescriptorData2::createData(ReSTIRGIBuffer[!id] , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 18 , 0);

                    data[19] = DescriptorData2::createData(skybox , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 19 , 0);
                    data[20] = DescriptorData2::createData(sampler.get() , 20 , 0);
                    data[21] = DescriptorData2::createData(transmissionLUT , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 21 , 0);
                    data[22] = DescriptorData2::createData(multiScatter , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 22 , 0);

                    data[23] = DescriptorData2::createData(diffuseImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 23 , 0);
                    data[24] = DescriptorData2::createData(specularImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 24 , 0);
                    data[25] = DescriptorData2::createData(albedoImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 25 , 0);
                    data[26] = DescriptorData2::createData(indirectImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 26 , 0);
                    data[27] = DescriptorData2::createData(normalImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 27 , 0);
                    data[28] = DescriptorData2::createData(positionImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 28 , 0);
                    data[29] = DescriptorData2::createData(roughnessImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 29 , 0);
                    data[30] = DescriptorData2::createData(metallicImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 30 , 0);
                    data[31] = DescriptorData2::createData(depthImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 31 , 0);
                    data[32] = DescriptorData2::createData(shadowImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 32 , 0);
                    data[33] = DescriptorData2::createData(shadowImage[!id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 33 , 0);
                    data[34] = DescriptorData2::createData(viewImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 34 , 0);
                    data[35] = DescriptorData2::createData(motionVector , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 35 , 0);
                    data[36] = DescriptorData2::createData(pixelCheck , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 36 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->addDescriptorBuffer2(info);
                }

                // ------------------------------------------------------------ the shaders
                ShaderModule rayGeneration(context , inputData.rayGenerationPath);

                ShaderModule generalMiss(context , inputData.generalMissPath.empty() ? inputData.giMissPath : inputData.generalMissPath);
                ShaderModule giMiss(context , inputData.giMissPath);
                ShaderModule shadowMiss(context , inputData.shadowMissPath);

                std::optional<ShaderModule> generalHit;
                if(!inputData.generalHitPath.empty()) generalHit.emplace(context , inputData.generalHitPath);
                ShaderModule giHit(context , inputData.giHitPath);
                ShaderModule shadowAnyHit(context , inputData.shadowAnyHitPath);

                // ------------------------------------------------------------ the stages and the groups
                // the SBT stores no names , only the order : THIS ORDER is what RT_MISS_* and RT_HITGROUP_* mean
                std::vector<VkPipelineShaderStageCreateInfo> stages{};
                stages.reserve(8);

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
                const uint32_t generalMissStage = addStage(VK_SHADER_STAGE_MISS_BIT_KHR , generalMiss.module);
                const uint32_t giMissStage = addStage(VK_SHADER_STAGE_MISS_BIT_KHR , giMiss.module);
                const uint32_t shadowMissStage = addStage(VK_SHADER_STAGE_MISS_BIT_KHR , shadowMiss.module);
                const uint32_t generalHitStage = generalHit.has_value() ? addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR , generalHit->module) : VK_SHADER_UNUSED_KHR;
                const uint32_t giHitStage = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR , giHit.module);
                const uint32_t shadowAnyHitStage = addStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR , shadowAnyHit.module);

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

                // a hit group of triangles : the closest hit and the any hit are both optional ( a group with none of them is an empty one )
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
                    generalGroup(generalMissStage) ,     // RT_MISS_GENERAL
                    generalGroup(giMissStage) ,          // RT_MISS_HITINFO
                    generalGroup(shadowMissStage) ,      // RT_MISS_SHADOW
                    hitGroup(generalHitStage , VK_SHADER_UNUSED_KHR) ,   // RT_HITGROUP_GENERAL
                    hitGroup(giHitStage , VK_SHADER_UNUSED_KHR) ,        // RT_HITGROUP_HITINFO
                    hitGroup(VK_SHADER_UNUSED_KHR , shadowAnyHitStage)   // RT_HITGROUP_SHADOW
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
                // the ray generation shader traces ( depth 1 ) and the hit shaders do not trace any more : the shadow rays and the GI rays both
                // start in the ray generation shader . a hit shader that traces needs more
                info.maxPipelineRayRecursionDepth = MAX_RAY_CAST;
                info.pLibraryInfo = nullptr;
                info.pLibraryInterface = nullptr;
                info.pDynamicState = nullptr;
                info.layout = pipe->layout->layout;
                info.basePipelineHandle = nullptr;
                info.basePipelineIndex = -1;

                const auto result = createRayTracingPipelinesKHR(context->logicalDevice->device , nullptr , nullptr , 1 , &info , nullptr , &pipe->pipeline);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the ray tracing pipeline [RayTracing] , error ID -> {}" , static_cast<int32_t>(result))});
                }

                // ------------------------------------------------------------ the shader binding table : the order of the groups is the one above
                {
                    SBT2::CreateInfo sbtInfo{};
                    sbtInfo.context = context;
                    sbtInfo.pipeline = pipe->pipeline;
                    sbtInfo.groupTypes = {
                        SBTGroupType::RayGeneration ,
                        SBTGroupType::Miss , SBTGroupType::Miss , SBTGroupType::Miss ,
                        SBTGroupType::Hit , SBTGroupType::Hit , SBTGroupType::Hit
                    };

                    sbt = std::make_shared<SBT2>(sbtInfo);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RayTracing pipe , please check the logs above"});
            }

            completedInit = true;
            return {};
        }
    };

    using pRayTracing = std::shared_ptr<RayTracing>;
}
