#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../AS2/AccelerationStructureBase.hpp"
#include "../Object2/Mesh2.hpp"
#include "../Sampler/Sampler.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../LightManager/LightManager.hpp"
#include "../NRCManager/NRCManager.hpp"

namespace mox{

    // =============================== INFO ===============================
    // NRCPass - the training of the Neural Radiance Cache : COMPUTE shaders ( the rays are the ray queries , not a ray tracing pipeline )
    //
    // the shaders of the pass ( the order = the index in pipe->shaders ) :
    //   0 : tracePath    - Shaders/NRC/trainingShader : the paths of the 4 vertices , the labelled samples go to NRCManager::trainingSampleBuffer
    //   1 : backwardPath - Shaders/NRC/backstepShader : the forward + the backward of the net , the gradients go to the grad buffers and to Parameter.grad of the hash grid
    //
    // the sets ( the same numbers in both shaders ) :
    //   set 0 : the list of RayTracing set 0 WITHOUT the images of the light ( diffuse 20 , specular 21 , indirect 23 : the training does not touch them ) , two descriptor
    //           buffers , the one with the index frameIndex % 2 is bound ( the ping pong images are swapped inside )
    //   set 1 : the textures of the TextureManager2 ( the alpha test of the rays )
    //   set 2 : the world grid of the lamps of the LightManager
    //   set 3 : the net of NRCManager ( NOT 4 like in RayTracing : there is no Radiance Cascade set here )
    //
    // what is NOT here yet : executePass ( the dispatches , the barriers between them ) , the apply pass ( Adam ) and the conversions of the matrices
    // =============================== END ===============================

    class NRCPass : public GraphPass2{
    public:

        SetTypicalOperators(NRCPass)
        SetTypicalTransferOperators(NRCPass)

        struct CreateInfo{
            VulkanContext* context{};
            AccelerationStructure* as{};
            TextureManager2* manager{};
            LightManager* lights{};
            NRC::NRCManager* nrc{};

            std::string tracePath;      // Shaders/NRC/trainingShader ( without the extension , like the other passes )
            std::string backwardPath;   // Shaders/NRC/backstepShader
            std::string applyPath;      // Shaders/NRC/applyShader ( Adam ) : only stored for now , createPipe does not load it yet
        };

        explicit NRCPass(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            if(!info.as){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create NRCPass , since the acceleration structure is nullptr"});
                THROW_MESSAGE;
            }

            if(!info.manager){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create NRCPass , since the texture manager is nullptr"});
                THROW_MESSAGE;
            }

            if(!info.lights){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create NRCPass , since the light manager is nullptr"});
                THROW_MESSAGE;
            }

            if(!info.nrc){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create NRCPass , since the NRC manager is nullptr"});
                THROW_MESSAGE;
            }

            if(info.tracePath.empty() || info.backwardPath.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create NRCPass , since one of the shader paths is empty ( trace , backward )"});
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~NRCPass() noexcept{
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs{};
            reqs.reserve(40);

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

            // the ping pong pairs ( no diffuse , specular , indirect : the training does not touch the light )
            image("positionImage_0" , ImageType::RenderImage2);
            image("positionImage_1" , ImageType::RenderImage2);
            image("normalImage_0" , ImageType::RenderImage2);
            image("normalImage_1" , ImageType::RenderImage2);
            image("depthImage_0" , ImageType::RenderImage2);
            image("depthImage_1" , ImageType::RenderImage2);
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
            buffer("ReSTIRGIBuffer_0");
            buffer("ReSTIRGIBuffer_1");
            buffer("tileLightIndices");
            buffer("lightCountPerTile");
            return reqs;
        }

        // ==================================================================================================================================================
        // TODO ( yours ) : the dispatches of the pass .
        //   what is ready : pipe->shaders[0]->shader ( trace ) , pipe->shaders[1]->shader ( backward ) , pipe->buffers[frameIndex % 2] ( the descriptor buffer of set 0 ) ,
        //   pipe->layout->layout , inputData.manager->dbuffer , inputData.lights->dbuffer , inputData.nrc->dbuffer ( the sets 1 , 2 , 3 ) .
        //   the pipeline layout is { set 0 , textures , lamps , net } , the bind point is VK_PIPELINE_BIND_POINT_COMPUTE ( see RayQueryShadows::executePass ) .
        //   the size of the dispatch is the half of the screen : NRC::GeneralData of NRCManager::generalDataBuffer ( width , height ) , the group is 8 x 8 .
        // ==================================================================================================================================================
        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [NRCPass]"});

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
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [NRCPass]"});

            const uint8_t id = inputData.context->sc->frameIndex % 2;
            const uint8_t prev = 1 - id;

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 write = VK_ACCESS_2_SHADER_WRITE_BIT;
            constexpr VkAccessFlags2 uniformRead = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            // the G buffer of this frame and the sky : only read ( the paths start from the G buffer of the pixel )
            std::array<VkImageMemoryBarrier2 , 15> imageBar;

            imageBar[0] = image("positionImage" , id)->translateImage(read , stage);
            imageBar[1] = image("normalImage" , id)->translateImage(read , stage);
            imageBar[2] = image("depthImage" , id)->translateImage(read , stage);
            imageBar[3] = image("albedoImage")->translateImage(read , stage);
            imageBar[4] = image("roughnessImage")->translateImage(read , stage);
            imageBar[5] = image("metallicImage")->translateImage(read , stage);
            imageBar[6] = image("motionVector")->translateImage(read , stage);
            imageBar[7] = image("pixelCheck")->translateImage(read , stage);

            // sampled : the sky , the transmittance of the sun and the LUT of the multi scattering
            imageBar[8] = skybox("skybox")->translateImage(read , stage);
            imageBar[9] = image("transmissionLUT")->translateImage(read , stage);
            imageBar[10] = image("multiScatter")->translateImage(read , stage);

            // the G buffer of the last frame
            imageBar[11] = image("positionImage" , prev)->translateImage(read , stage);
            imageBar[12] = image("normalImage" , prev)->translateImage(read , stage);
            imageBar[13] = image("depthImage" , prev)->translateImage(read , stage);

            imageBar[14] = image("objectAmbientImage")->translateImage(read , stage);

            std::array<VkBufferMemoryBarrier2 , 26> bufBar;

            bufBar[0] = buffer("generalBuffer")->translateBuffer(uniformRead , stage);
            bufBar[1] = buffer("generalFeatures")->translateBuffer(uniformRead , stage);
            bufBar[2] = buffer("forwardInfo")->translateBuffer(uniformRead , stage);
            bufBar[3] = buffer("skyBuffer")->translateBuffer(uniformRead , stage);
            bufBar[4] = buffer("sunBuffer")->translateBuffer(uniformRead , stage);

            // the world grid of the lamps
            bufBar[5] = inputData.lights->lightBuffer->translateBuffer(read , stage);
            bufBar[6] = inputData.lights->lightCellBuffer->translateBuffer(read , stage);
            bufBar[7] = inputData.lights->lightCascadeBuffer->translateBuffer(uniformRead , stage);

            // the geometry and the materials : the ray queries read them
            bufBar[8] = Mesh2::vertexBuffer->translateBuffer(read , stage);
            bufBar[9] = Mesh2::indicesBuffer->translateBuffer(read , stage);
            bufBar[10] = Mesh2::instanceLayoutBuffer->translateBuffer(read , stage);
            bufBar[11] = Mesh2::instanceBuffer->translateBuffer(read , stage);
            bufBar[12] = Mesh2::MeshBuffer->translateBuffer(read , stage);
            bufBar[13] = Mesh2::LUT_buffer->translateBuffer(read , stage);

            bufBar[14] = buffer("tileLightIndices")->translateBuffer(read , stage);
            bufBar[15] = buffer("lightCountPerTile")->translateBuffer(read , stage);

            // the net : the trace pass writes the samples , the backward pass reads them and adds into the gradients ( the matrices and the hash grid ) , the weights are only read
            bufBar[16] = inputData.nrc->trainingSampleBuffer->translateBuffer(read | write , stage);
            bufBar[17] = inputData.nrc->generalDataBuffer->translateBuffer(read | write , stage);
            bufBar[18] = inputData.nrc->hashGridBuffer->translateBuffer(read , stage);
            bufBar[19] = inputData.nrc->hashGridParamsBuffer->translateBuffer(read | write , stage);
            bufBar[20] = inputData.nrc->hashGridGeneralBuffer->translateBuffer(read , stage);
            bufBar[21] = inputData.nrc->neuronHiddenBuffer->translateBuffer(read , stage);
            bufBar[22] = inputData.nrc->neuronPerChannelBuffer->translateBuffer(read , stage);
            bufBar[23] = inputData.nrc->neuronHiddenGradBuffer->translateBuffer(read | write , stage);
            bufBar[24] = inputData.nrc->neuronPerChannelGradBuffer->translateBuffer(read | write , stage);
            bufBar[25] = inputData.nrc->pointersBuffer->translateBuffer(uniformRead , stage);

            auto dep = TypicalDependency<26 , 15>(imageBar.data() , bufBar.data());

            vkCmdPipelineBarrier2(cmd , &dep);

            return {};
        }

    private:

        pShaderPipeType pipe{nullptr};
        pSampler sampler{nullptr};

        CreateInfo inputData{};

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            VulkanContext* context = inputData.context;

            try{
                pipe = std::make_shared<ShaderPipeType>(context);

                {
                    Sampler::CreateInfo info{};
                    info.context = context;
                    info.linearClampSampler = true;

                    sampler = std::make_shared<Sampler>(info);
                }

                // ------------------------------------------------------------ set 0 : the layout of RayTracing , but for the COMPUTE stage
                {
                    std::vector<DescriptorLayoutData> data(40);

                    data[0] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR , 1};

                    for(uint32_t i = 1 ; i <= 5 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    for(uint32_t i = 6 ; i <= 15 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    // 12 and 13 were the reservoirs of ReSTIR DI : reserved ( no descriptors ) , the other bindings keep their numbers
                    data[12] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0};
                    data[13] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0};

                    data[16] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[17] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[18] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[19] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    for(uint32_t i = 20 ; i <= 33 ; i++) data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    // the images of the light : the shaders of the training do not declare them
                    data[20] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0};
                    data[21] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0};
                    data[23] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0};

                    data[34] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[35] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    data[36] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[37] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[38] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    data[39] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                    pipe->addDescriptorLayout(info);
                }

                // ------------------------------------------------------------ the pipeline layout : set 0 , the textures , the world grid of the lamps , the net
                {
                    PipelineLayout::CreateInfo info{};
                    info.context = context;
                    info.layouts = {pipe->descriptorLayouts[0].get() , inputData.manager->layout.get() , inputData.lights->layout.get() , inputData.nrc->layout.get()};
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                // ------------------------------------------------------------ set 0 : the descriptor buffers , one per parity of the frame
                for(const auto id : {0 , 1}){
                    std::vector<DescriptorData2> data(40);

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

                    data[14] = DescriptorData2::createData(buffer("ReSTIRGIBuffer" , id) , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 14 , 0);
                    data[15] = DescriptorData2::createData(buffer("ReSTIRGIBuffer" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 15 , 0);

                    data[16] = DescriptorData2::createData(skybox("skybox") , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 16 , 0);
                    data[17] = DescriptorData2::createData(sampler.get() , 17 , 0);
                    data[18] = DescriptorData2::createData(image("transmissionLUT") , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 18 , 0);
                    data[19] = DescriptorData2::createData(image("multiScatter") , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 19 , 0);

                    data[22] = DescriptorData2::createData(image("albedoImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 22 , 0);
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

                    // the bindings with no descriptors ( 12 , 13 : ReSTIR DI , 20 , 21 , 23 : the images of the light ) are not written : from the end to the start , the numbers do not move
                    data.erase(data.begin() + 23);
                    data.erase(data.begin() + 20 , data.begin() + 22);
                    data.erase(data.begin() + 12 , data.begin() + 14);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->addDescriptorBuffer2(info);
                }

                // ------------------------------------------------------------ the shaders : the same four sets in both
                for(const std::string &path : {inputData.tracePath , inputData.backwardPath}){
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = path;
                    info.layouts = {pipe->descriptorLayouts[0].get() , inputData.manager->layout.get() , inputData.lights->layout.get() , inputData.nrc->layout.get()};
                    info.pushes = {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create NRCPass , please check the logs above"});
            }

            completedInit = true;
            return {};
        }

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(NRCPass&& input) noexcept{
            if(this == &input) return;
            processDelete();
            GraphPass2::operator=(std::move(input));

            pipe = std::move(input.pipe);
            sampler = std::move(input.sampler);
            // the acceleration structure and the managers are not owned by the pass
            inputData = input.inputData;

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            clearAttributes();
            // the images and the buffers belong to the graph , they are only forgotten
            pipe = nullptr;
            sampler = nullptr;

            inputData = {};
            completedInit = false;
        }
    };

    using pNRCPass = std::shared_ptr<NRCPass>;
}
