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
#include "../LightManager/LightManager.hpp"

namespace mox{

    // =============================== INFO ===============================
    // PathTracing - the path tracing pass that runs AFTER the Deferred render : the primary rays start in the ray generation shader , but the surface of the primary hit is taken
    // from the G buffer of the raster ( it has the mip levels of the textures ) . the closest hit of the primary ray writes the lighting and the history of the reservoirs ,
    // the miss writes the sky , the bounces start in the closest hit of the primary ray
    //
    // the pipeline is a REAL ray tracing pipeline ( vkCreateRayTracingPipelinesKHR ) , its descriptors are descriptor buffers :
    //   set 0 : 42 bindings , the same list as in every shader of Shaders/PathTracing , two descriptor buffers , the one with the index frameIndex % 2 is bound ( the ping pong
    //           images and the reservoirs are swapped inside )
    //   set 1 : the textures of the TextureManager2 ( the same set as the Deferred render )
    //   set 2 : the world grid of the lamps of the LightManager
    //
    // the order of the groups = the numbers in shaderStructure.h ( RT_MISS_* and RT_HITGROUP_* ) :
    //   miss : 0 the primary ray ( the sky ) , 1 the secondary ray ( bounces ) , 2 shadow
    //   hit  : 0 the primary ray , 1 the secondary ray ( bounces ) , 2 shadow ( the any hit , and the closest hit when shadowHitPath is not empty )
    //
    // NO pointers to the images and the buffers of the graph here : everything that the pass requires lives in the attributes of GraphPass2 ( image( "name" ) , buffer( "name" ) )
    // =============================== END ===============================

    class PathTracing : public GraphPass2{
    public:

        SetTypicalOperators(PathTracing)
        SetTypicalTransferOperators(PathTracing)

        struct CreateInfo{
            VulkanContext* context{};
            AccelerationStructure* as{};
            TextureManager2* manager{};
            // the layout and the descriptor buffer of the set 2 ( the world grid of the lamps ) : the manager has to live longer than the pass
            LightManager* lights{};

            std::string rayGenerationPath;

            // miss shaders
            std::string primaryMissPath;     // RT_MISS_GENERAL : the primary ray left the scene
            std::string secondaryMissPath;   // RT_MISS_HITINFO : a bounce left the scene
            std::string shadowMissPath;      // RT_MISS_SHADOW

            // hit shaders
            std::string primaryHitPath;      // RT_HITGROUP_GENERAL : the closest hit of the primary ray
            std::string secondaryHitPath;    // RT_HITGROUP_HITINFO : the closest hit of a bounce
            std::string shadowAnyHitPath;    // RT_HITGROUP_SHADOW : the any hit
            std::string shadowHitPath;       // RT_HITGROUP_SHADOW , optional : the closest hit ( the group is an any hit only when it is empty )

            // the any hit shaders of the alpha test ( glTF MASK ) : the holes of the leaves are not hit . optional , without them the masked materials are opaque for the rays .
            // the payload of an any hit has to be the one of the closest hit of its group
            std::string primaryAlphaAnyHitPath;     // RT_HITGROUP_GENERAL
            std::string secondaryAlphaAnyHitPath;   // RT_HITGROUP_HITINFO
        };

        explicit PathTracing(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            if(!info.as){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create PathTracing , since the acceleration structure is nullptr"});
                THROW_MESSAGE;
            }

            if(!info.manager){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create PathTracing , since the texture manager is nullptr"});
                THROW_MESSAGE;
            }

            if(!info.lights){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create PathTracing , since the light manager is nullptr"});
                THROW_MESSAGE;
            }

            if(info.rayGenerationPath.empty() || info.primaryMissPath.empty() || info.secondaryMissPath.empty() || info.shadowMissPath.empty() ||
               info.primaryHitPath.empty() || info.secondaryHitPath.empty() || info.shadowAnyHitPath.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create PathTracing , since one of the required shader paths is empty ( ray generation , primary / secondary / shadow miss , primary / secondary hit , shadow any hit )"});
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~PathTracing() noexcept{
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
            image("objectAmbientImage" , ImageType::RenderImage2);
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
            buffer("ReSTIRDIBuffer_0");
            buffer("ReSTIRDIBuffer_1");
            buffer("ReSTIRGIBuffer_0");
            buffer("ReSTIRGIBuffer_1");
            buffer("ReSTIRPTBuffer_0");
            buffer("ReSTIRPTBuffer_1");
            buffer("tileLightIndices");
            buffer("lightCountPerTile");
            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [PathTracing]"});

            const uint8_t currentImage = inputData.context->sc->frameIndex % 2;

            vkCmdBindPipeline(cmd , VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR , pipe->pipeline);

            // set 0 : the descriptor buffer of this pass , set 1 : the textures , set 2 : the world grid of the lamps
            {
                std::array<VkDescriptorBufferBindingInfoEXT , 3> info{};

                info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[0].address = pipe->buffers[currentImage]->pBuffer;
                info[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[0].pNext = nullptr;

                info[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[1].address = inputData.manager->dbuffer->pBuffer;
                info[1].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[1].pNext = nullptr;

                info[2].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[2].address = inputData.lights->dbuffer->pBuffer;
                info[2].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[2].pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 3 , info.data());

                const uint32_t sets[3] = {0 , 1 , 2};
                const VkDeviceSize offsets[3] = {0 , 0 , 0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR , pipe->layout->layout , 0 , 3 , sets , offsets);
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

            const auto bound = bindAttributes(sources);
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [PathTracing]"});

            const uint8_t id = inputData.context->sc->frameIndex % 2;
            const uint8_t prev = 1 - id;

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 write = VK_ACCESS_2_SHADER_WRITE_BIT;
            constexpr VkAccessFlags2 uniformRead = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            // the G buffer and the light of THIS frame are written by the closest hit and the miss ( and the closest hit reads the position and the normal back : checkDisocclusion ) :
            // read and write . the G buffer of the last frame and the sky : only read
            std::array<VkImageMemoryBarrier2 , 21> imageBar;

            imageBar[0] = image("positionImage" , id)->translateImage(read | write , stage);
            imageBar[1] = image("normalImage" , id)->translateImage(read | write , stage);
            imageBar[2] = image("depthImage" , id)->translateImage(read | write , stage);
            imageBar[3] = image("albedoImage")->translateImage(read | write , stage);
            imageBar[4] = image("roughnessImage")->translateImage(read | write , stage);
            imageBar[5] = image("metallicImage")->translateImage(read | write , stage);
            imageBar[6] = image("objectAmbientImage")->translateImage(read | write , stage);
            imageBar[7] = image("motionVector")->translateImage(read | write , stage);
            imageBar[8] = image("pixelCheck")->translateImage(read | write , stage);
            imageBar[9] = image("viewImage")->translateImage(read | write , stage);

            imageBar[10] = image("diffuseImage" , id)->translateImage(read | write , stage);
            imageBar[11] = image("specularImage" , id)->translateImage(read | write , stage);
            imageBar[12] = image("indirectImage" , id)->translateImage(read | write , stage);
            imageBar[13] = image("shadowImage" , id)->translateImage(read | write , stage);

            // the history of the last frame : only read
            imageBar[14] = image("shadowImage" , prev)->translateImage(read , stage);
            imageBar[15] = image("positionImage" , prev)->translateImage(read , stage);
            imageBar[16] = image("normalImage" , prev)->translateImage(read , stage);
            imageBar[17] = image("depthImage" , prev)->translateImage(read , stage);

            // sampled : the sky , the transmittance of the sun and the LUT of the multi scattering
            imageBar[18] = skybox("skybox")->translateImage(read , stage);
            imageBar[19] = image("transmissionLUT")->translateImage(read , stage);
            imageBar[20] = image("multiScatter")->translateImage(read , stage);

            std::array<VkBufferMemoryBarrier2 , 22> bufBar;

            bufBar[0] = buffer("generalBuffer")->translateBuffer(uniformRead , stage);
            bufBar[1] = buffer("generalFeatures")->translateBuffer(uniformRead , stage);
            bufBar[2] = buffer("forwardInfo")->translateBuffer(uniformRead , stage);
            bufBar[3] = buffer("skyBuffer")->translateBuffer(uniformRead , stage);
            bufBar[4] = buffer("sunBuffer")->translateBuffer(uniformRead , stage);

            // the world grid of the lamps : written by LightManager::updateGrid ( the compute shader ) , the buffers belong to the manager
            bufBar[5] = inputData.lights->lightBuffer->translateBuffer(read , stage);
            bufBar[6] = inputData.lights->lightCellBuffer->translateBuffer(read , stage);
            bufBar[7] = inputData.lights->lightCascadeBuffer->translateBuffer(uniformRead , stage);

            // the geometry and the materials : the closest hit reads them
            bufBar[8] = Mesh2::vertexBuffer->translateBuffer(read , stage);
            bufBar[9] = Mesh2::indicesBuffer->translateBuffer(read , stage);
            bufBar[10] = Mesh2::instanceLayoutBuffer->translateBuffer(read , stage);
            bufBar[11] = Mesh2::instanceBuffer->translateBuffer(read , stage);
            bufBar[12] = Mesh2::MeshBuffer->translateBuffer(read , stage);
            bufBar[13] = Mesh2::LUT_buffer->translateBuffer(read , stage);

            // the reservoirs : this frame writes its own ( and reads it back ) and reads the ones of the previous frame
            bufBar[14] = buffer("ReSTIRDIBuffer" , id)->translateBuffer(read | write , stage);
            bufBar[15] = buffer("ReSTIRDIBuffer" , prev)->translateBuffer(read , stage);
            bufBar[16] = buffer("ReSTIRGIBuffer" , id)->translateBuffer(read | write , stage);
            bufBar[17] = buffer("ReSTIRGIBuffer" , prev)->translateBuffer(read , stage);

            // the lists of the lamps of the tiles : written by the ForwardPlus
            bufBar[18] = buffer("tileLightIndices")->translateBuffer(read , stage);
            bufBar[19] = buffer("lightCountPerTile")->translateBuffer(read , stage);

            // the reservoirs of ReSTIR PT : this frame writes its own ( and reads it back ) and reads the one of the previous frame
            bufBar[20] = buffer("ReSTIRPTBuffer" , id)->translateBuffer(read | write , stage);
            bufBar[21] = buffer("ReSTIRPTBuffer" , prev)->translateBuffer(read , stage);

            auto dep = TypicalDependency<22 , 21>(imageBar.data() , bufBar.data());

            vkCmdPipelineBarrier2(cmd , &dep);

            return {};
        }

    private:

        pPipeline pipe{nullptr};
        pSBT2 sbt{nullptr};
        pSampler sampler{nullptr};

        CreateInfo inputData{};

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(PathTracing&& input) noexcept{
            if(this == &input) return;
            processDelete();
            GraphPass2::operator=(std::move(input));

            pipe = std::move(input.pipe);
            sbt = std::move(input.sbt);
            sampler = std::move(input.sampler);
            // the acceleration structure , the texture manager and the light manager are not owned by the pass
            inputData = input.inputData;

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            clearAttributes();
            // the images and the buffers belong to the graph , they are only forgotten . the SBT is destroyed before the pipeline it was made from
            sbt = nullptr;
            pipe = nullptr;
            sampler = nullptr;

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
                    std::vector<DescriptorLayoutData> data(42);

                    data[0] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR , 1};

                    for(uint32_t i = 1 ; i <= 5 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    for(uint32_t i = 6 ; i <= 15 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    data[16] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[17] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[18] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[19] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    for(uint32_t i = 20 ; i <= 33 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    data[34] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[35] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    // the G buffer of the LAST frame ( position , normal , depth ) : the history of the temporal part checks with it that the surface is the same
                    data[36] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[37] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[38] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    // the occlusion of the material ( the ao map of the object ) : the closest hit writes it , the direct light is multiplied by it
                    data[39] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    // the reservoirs of ReSTIR PT : this frame ( 40 ) and the last frame ( 41 )
                    data[40] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[41] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

                    pipe->addDescriptorLayout(info);
                }

                // ------------------------------------------------------------ the pipeline layout : set 0 , the textures and the world grid of the lamps
                {
                    PipelineLayout::CreateInfo info{};
                    info.context = context;
                    info.layouts = {pipe->descriptorLayouts[0].get() , inputData.manager->layout.get() , inputData.lights->layout.get()};
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                // ------------------------------------------------------------ set 0 : the descriptor buffers , one per parity of the frame
                for(const auto id : {0 , 1}){
                    std::vector<DescriptorData2> data(42);

                    data[0] = DescriptorData2::createData(inputData.as , 0 , 0);

                    data[1] = DescriptorData2::createData(buffer("generalBuffer") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0);
                    data[2] = DescriptorData2::createData(buffer("forwardInfo") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0);
                    data[3] = DescriptorData2::createData(buffer("skyBuffer") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 3 , 0);
                    data[4] = DescriptorData2::createData(buffer("generalFeatures") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 4 , 0);
                    data[5] = DescriptorData2::createData(buffer("sunBuffer") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 5 , 0);

                    data[6] = DescriptorData2::createData(Mesh2::vertexBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 6 , 0);
                    data[7] = DescriptorData2::createData(Mesh2::indicesBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 7 , 0);
                    data[8] = DescriptorData2::createData(Mesh2::instanceLayoutBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 8 , 0);
                    data[9] = DescriptorData2::createData(Mesh2::instanceBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 9 , 0);
                    data[10] = DescriptorData2::createData(Mesh2::MeshBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 10 , 0);
                    data[11] = DescriptorData2::createData(Mesh2::LUT_buffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 11 , 0);

                    // the reservoirs : this frame writes the one with its own parity and reads the other
                    data[12] = DescriptorData2::createData(buffer("ReSTIRDIBuffer" , id) , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 12 , 0);
                    data[13] = DescriptorData2::createData(buffer("ReSTIRDIBuffer" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 13 , 0);
                    data[14] = DescriptorData2::createData(buffer("ReSTIRGIBuffer" , id) , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 14 , 0);
                    data[15] = DescriptorData2::createData(buffer("ReSTIRGIBuffer" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 15 , 0);

                    data[16] = DescriptorData2::createData(skybox("skybox") , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 16 , 0);
                    data[17] = DescriptorData2::createData(sampler.get() , 17 , 0);
                    data[18] = DescriptorData2::createData(image("transmissionLUT") , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 18 , 0);
                    data[19] = DescriptorData2::createData(image("multiScatter") , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 19 , 0);

                    data[20] = DescriptorData2::createData(image("diffuseImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 20 , 0);
                    data[21] = DescriptorData2::createData(image("specularImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 21 , 0);
                    data[22] = DescriptorData2::createData(image("albedoImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 22 , 0);
                    data[23] = DescriptorData2::createData(image("indirectImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 23 , 0);
                    data[24] = DescriptorData2::createData(image("normalImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 24 , 0);
                    data[25] = DescriptorData2::createData(image("positionImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 25 , 0);
                    data[26] = DescriptorData2::createData(image("roughnessImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 26 , 0);
                    data[27] = DescriptorData2::createData(image("metallicImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 27 , 0);
                    data[28] = DescriptorData2::createData(image("depthImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 28 , 0);
                    data[29] = DescriptorData2::createData(image("shadowImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 29 , 0);
                    data[30] = DescriptorData2::createData(image("shadowImage" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 30 , 0);
                    data[31] = DescriptorData2::createData(image("viewImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 31 , 0);
                    data[32] = DescriptorData2::createData(image("motionVector") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 32 , 0);
                    data[33] = DescriptorData2::createData(image("pixelCheck") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 33 , 0);
                    data[34] = DescriptorData2::createData(buffer("lightCountPerTile") , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 34 , 0);
                    data[35] = DescriptorData2::createData(buffer("tileLightIndices") , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 35 , 0);

                    // the images of the other parity ( !id ) : the G buffer of the last frame
                    data[36] = DescriptorData2::createData(image("positionImage" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 36 , 0);
                    data[37] = DescriptorData2::createData(image("normalImage" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 37 , 0);
                    data[38] = DescriptorData2::createData(image("depthImage" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 38 , 0);

                    data[39] = DescriptorData2::createData(image("objectAmbientImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 39 , 0);

                    // the reservoirs of ReSTIR PT : this frame writes the one with its own parity and reads the other
                    data[40] = DescriptorData2::createData(buffer("ReSTIRPTBuffer" , id) , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 40 , 0);
                    data[41] = DescriptorData2::createData(buffer("ReSTIRPTBuffer" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 41 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->addDescriptorBuffer2(info);
                }

                // ------------------------------------------------------------ the shaders
                ShaderModuleType rayGeneration(context , inputData.rayGenerationPath);

                ShaderModuleType primaryMiss(context , inputData.primaryMissPath);
                ShaderModuleType secondaryMiss(context , inputData.secondaryMissPath);
                ShaderModuleType shadowMiss(context , inputData.shadowMissPath);

                ShaderModuleType primaryHit(context , inputData.primaryHitPath);
                ShaderModuleType secondaryHit(context , inputData.secondaryHitPath);
                ShaderModuleType shadowAnyHit(context , inputData.shadowAnyHitPath);
                std::optional<ShaderModuleType> shadowHit;
                if(!inputData.shadowHitPath.empty()) shadowHit.emplace(context , inputData.shadowHitPath);
                std::optional<ShaderModuleType> primaryAlphaAnyHit;
                if(!inputData.primaryAlphaAnyHitPath.empty()) primaryAlphaAnyHit.emplace(context , inputData.primaryAlphaAnyHitPath);
                std::optional<ShaderModuleType> secondaryAlphaAnyHit;
                if(!inputData.secondaryAlphaAnyHitPath.empty()) secondaryAlphaAnyHit.emplace(context , inputData.secondaryAlphaAnyHitPath);

                // ------------------------------------------------------------ the stages and the groups
                // the SBT stores no names , only the order : THIS ORDER is what RT_MISS_* and RT_HITGROUP_* mean
                std::vector<VkPipelineShaderStageCreateInfo> stages{};
                stages.reserve(10);

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
                const uint32_t primaryMissStage = addStage(VK_SHADER_STAGE_MISS_BIT_KHR , primaryMiss.module);
                const uint32_t secondaryMissStage = addStage(VK_SHADER_STAGE_MISS_BIT_KHR , secondaryMiss.module);
                const uint32_t shadowMissStage = addStage(VK_SHADER_STAGE_MISS_BIT_KHR , shadowMiss.module);
                const uint32_t primaryHitStage = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR , primaryHit.module);
                const uint32_t secondaryHitStage = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR , secondaryHit.module);
                const uint32_t shadowAnyHitStage = addStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR , shadowAnyHit.module);
                const uint32_t shadowHitStage = shadowHit.has_value() ? addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR , shadowHit->module) : VK_SHADER_UNUSED_KHR;
                const uint32_t primaryAlphaStage = primaryAlphaAnyHit.has_value() ? addStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR , primaryAlphaAnyHit->module) : VK_SHADER_UNUSED_KHR;
                const uint32_t secondaryAlphaStage = secondaryAlphaAnyHit.has_value() ? addStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR , secondaryAlphaAnyHit->module) : VK_SHADER_UNUSED_KHR;

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
                    generalGroup(primaryMissStage) ,     // RT_MISS_GENERAL
                    generalGroup(secondaryMissStage) ,   // RT_MISS_HITINFO
                    generalGroup(shadowMissStage) ,      // RT_MISS_SHADOW
                    hitGroup(primaryHitStage , primaryAlphaStage) ,      // RT_HITGROUP_GENERAL
                    hitGroup(secondaryHitStage , secondaryAlphaStage) ,  // RT_HITGROUP_HITINFO
                    hitGroup(shadowHitStage , shadowAnyHitStage)         // RT_HITGROUP_SHADOW
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
                // the recursion that the device can do : the pipeline can not ask for more than this
                uint32_t deviceRecursionLimit = 1u;
                {
                    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{};
                    rtProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
                    VkPhysicalDeviceProperties2 properties2{};
                    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                    properties2.pNext = &rtProperties;
                    vkGetPhysicalDeviceProperties2(context->physicalDevice->device , &properties2);
                    deviceRecursionLimit = rtProperties.maxRayRecursionDepth;
                }
                // the ray generation shader traces the primary ray ( the hit is the level 1 ) , every bounce is one level more , and the shadow ray of the last bounce is one more again :
                // MAX_PT_RAY_CAST bounces need MAX_PT_RAY_CAST + 2 ( not more than the device can do )
                info.maxPipelineRayRecursionDepth = std::min(static_cast<uint32_t>(MAX_PT_RAY_CAST) + 2u , deviceRecursionLimit);
                info.pLibraryInfo = nullptr;
                info.pLibraryInterface = nullptr;
                info.pDynamicState = nullptr;
                info.layout = pipe->layout->layout;
                info.basePipelineHandle = nullptr;
                info.basePipelineIndex = -1;

                const auto result = createRayTracingPipelinesKHR(context->logicalDevice->device , nullptr , nullptr , 1 , &info , nullptr , &pipe->pipeline);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the ray tracing pipeline [PathTracing] , error ID -> {}" , static_cast<int32_t>(result))});
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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create PathTracing pipe , please check the logs above"});
            }

            completedInit = true;
            return {};
        }
    };

    using pPathTracing = std::shared_ptr<PathTracing>;
}
