#pragma once

#include "../ShaderPipe/ShaderPipe.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Sampler/Sampler.hpp"



namespace mox{

    class LightPass : public GraphPass2{
    public:

        SetTypicalOperators(LightPass)
        SetTypicalTransferOperators(LightPass)

        struct CreateInfo{
            VulkanContext* context;
            std::string path;
        };

        explicit LightPass(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            if(info.path.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create LightPass , since the path to the shader is empty"});
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~LightPass(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs{};
            reqs.reserve(30);

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

            // the ping-pong pairs
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

            image("pixelCheck" , ImageType::RenderImage2);
            image("albedoImage" , ImageType::RenderImage2);
            image("roughnessImage" , ImageType::RenderImage2);
            image("metallicImage" , ImageType::RenderImage2);
            image("filteredShadowImage" , ImageType::RenderImage2);
            image("filteredAmbientOcclusion" , ImageType::RenderImage2);
            image("motionVector" , ImageType::RenderImage2);
            image("multiScatter" , ImageType::RenderImage2);
            image("skybox" , ImageType::Skybox2);
            image("transmissionLUT" , ImageType::RenderImage2);

            buffer("generalBuffer");
            buffer("generalFeatures");
            buffer("forwardInfo");
            buffer("tileLightIndices");
            buffer("lightCountPerTile");
            buffer("lightBuffer");
            buffer("ReSTIRDIBuffer_0");
            buffer("ReSTIRDIBuffer_1");
            buffer("sunBuffer");
            buffer("skyBuffer");

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [LightPass]"});

            const uint8_t currentImage = inputData.context->sc->frameIndex % 2;

            const uint32_t x = (inputData.context->sc->extent.width + 7) / 8;
            const uint32_t y = (inputData.context->sc->extent.height + 7) / 8;

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.address = pipe->buffers[currentImage]->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 1,  &info);

                uint32_t sets[1] = {0};
                VkDeviceSize offset[] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offset);
            }

            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);

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
                GraphSlot<RenderImage2>{"pixelCheck" , &pixelCheck} , GraphSlot<RenderImage2>{"albedoImage" , &albedoImage} ,
                GraphSlot<RenderImage2>{"roughnessImage" , &roughnessImage} , GraphSlot<RenderImage2>{"metallicImage" , &metallicImage} ,
                GraphSlot<RenderImage2>{"filteredShadowImage" , &filteredShadowImage} , GraphSlot<RenderImage2>{"filteredAmbientOcclusion" , &filteredAmbientOcclusion} ,
                GraphSlot<RenderImage2>{"motionVector" , &motionVector} ,
                GraphSlot<RenderImage2>{"multiScatter" , &multiScatter} , GraphSlot<Skybox2>{"skybox" , &skybox} , GraphSlot<RenderImage2>{"transmissionLUT" , &transmissionLUT} ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer} , GraphSlot<BufferBase2>{"generalFeatures" , &features} ,
                GraphSlot<BufferBase2>{"forwardInfo" , &forwardInfo} , GraphSlot<BufferBase2>{"tileLightIndices" , &tileLightIndices} ,
                GraphSlot<BufferBase2>{"lightCountPerTile" , &lightCountPerTile} , GraphSlot<BufferBase2>{"lightBuffer" , &lightBuffer} ,
                GraphSlot<BufferBase2>{"ReSTIRDIBuffer_0" , &ReSTIRDIBuffer[0]} , GraphSlot<BufferBase2>{"ReSTIRDIBuffer_1" , &ReSTIRDIBuffer[1]} ,
                GraphSlot<BufferBase2>{"sunBuffer" , &sunBuffer} , GraphSlot<BufferBase2>{"skyBuffer" , &skyBuffer});
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [LightPass]"});

            const uint8_t id = inputData.context->sc->frameIndex % 2;
            const uint8_t prev = 1 - id;

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 write = VK_ACCESS_2_SHADER_WRITE_BIT;
            constexpr VkAccessFlags2 uniformRead = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            std::array<VkImageMemoryBarrier2 , 15> imageBar;

            // the g buffer of this frame , the shadows and the occlusion : only read
            imageBar[0] = positionImage[id]->translateImage(read , stage);
            imageBar[1] = normalImage[id]->translateImage(read , stage);
            imageBar[2] = depthImage[id]->translateImage(read , stage);
            imageBar[3] = albedoImage->translateImage(read , stage);
            imageBar[4] = roughnessImage->translateImage(read , stage);
            imageBar[5] = metallicImage->translateImage(read , stage);
            imageBar[6] = filteredShadowImage->translateImage(read , stage);
            imageBar[7] = filteredAmbientOcclusion->translateImage(read , stage);
            // sampled : the LUT of the multi scattering and the sky
            imageBar[8] = multiScatter->translateImage(read , stage);
            imageBar[9] = skybox->translateImage(read , stage);

            // the pixels of the sky are marked in the pixel check
            imageBar[10] = pixelCheck->translateImage(read | write , stage);

            // the result
            imageBar[11] = diffuseImage[id]->translateImage(write , stage);
            imageBar[12] = specularImage[id]->translateImage(write , stage);
            imageBar[13] = motionVector->translateImage(write , stage);

            // the sky : the transmittance of the sun disc
            imageBar[14] = transmissionLUT->translateImage(read , stage);

            std::array<VkBufferMemoryBarrier2 , 10> bufBar;

            bufBar[0] = generalBuffer->translateBuffer(uniformRead , stage);
            bufBar[1] = features->translateBuffer(uniformRead , stage);
            bufBar[2] = forwardInfo->translateBuffer(uniformRead , stage);
            // the lists of the lights : written by the ForwardPlus
            bufBar[3] = tileLightIndices->translateBuffer(read , stage);
            bufBar[4] = lightCountPerTile->translateBuffer(read , stage);
            bufBar[5] = lightBuffer->translateBuffer(read , stage);
            // the reservoirs : this frame writes its own and reads the ones of the previous frame
            bufBar[6] = ReSTIRDIBuffer[id]->translateBuffer(write , stage);
            bufBar[7] = ReSTIRDIBuffer[prev]->translateBuffer(read , stage);
            // the sun disc
            bufBar[8] = sunBuffer->translateBuffer(uniformRead , stage);
            bufBar[9] = skyBuffer->translateBuffer(uniformRead , stage);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = static_cast<uint32_t>(bufBar.size());
            info.imageMemoryBarrierCount = static_cast<uint32_t>(imageBar.size());
            info.pMemoryBarriers = nullptr;
            info.pBufferMemoryBarriers = bufBar.data();
            info.pImageMemoryBarriers = imageBar.data();
            info.memoryBarrierCount = 0;
            info.pNext = nullptr;
            info.dependencyFlags = 0;

            vkCmdPipelineBarrier2(cmd , &info);

            return {};
        }

    private:

        pShaderPipe pipe{nullptr};

        CreateInfo inputData{};

        // typicalSampler of the shader : the multi scattering LUT and the sky are filtered
        pSampler sampler{nullptr};

        RenderImage2* positionImage[2]{nullptr , nullptr};
        RenderImage2* normalImage[2]{nullptr , nullptr};
        RenderImage2* depthImage[2]{nullptr , nullptr};
        RenderImage2* diffuseImage[2]{nullptr , nullptr};
        RenderImage2* specularImage[2]{nullptr , nullptr};
        // 10
        RenderImage2* pixelCheck{nullptr};
        RenderImage2* albedoImage{nullptr};
        RenderImage2* roughnessImage{nullptr};
        RenderImage2* metallicImage{nullptr};
        RenderImage2* filteredShadowImage{nullptr};
        RenderImage2* filteredAmbientOcclusion{nullptr};
        RenderImage2* motionVector{nullptr};
        RenderImage2* multiScatter{nullptr};
        RenderImage2* transmissionLUT{nullptr};
        // 19
        Skybox2* skybox{nullptr};
        // 20
        BufferBase2* generalBuffer{nullptr};
        BufferBase2* features{nullptr};
        BufferBase2* forwardInfo{nullptr};
        BufferBase2* tileLightIndices{nullptr};
        BufferBase2* lightCountPerTile{nullptr};
        BufferBase2* lightBuffer{nullptr};
        BufferBase2* ReSTIRDIBuffer[2]{nullptr , nullptr};
        BufferBase2* sunBuffer{nullptr};
        BufferBase2* skyBuffer{nullptr};
        // 30

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(LightPass&& input) noexcept{
            if(this == &input) return;
            processDelete();

            pipe = std::move(input.pipe);
            inputData = input.inputData;
            sampler = std::move(input.sampler);

            for(const auto i : {0 , 1}){
                positionImage[i] = std::exchange(input.positionImage[i] , nullptr);
                normalImage[i] = std::exchange(input.normalImage[i] , nullptr);
                depthImage[i] = std::exchange(input.depthImage[i] , nullptr);
                diffuseImage[i] = std::exchange(input.diffuseImage[i] , nullptr);
                specularImage[i] = std::exchange(input.specularImage[i] , nullptr);
                ReSTIRDIBuffer[i] = std::exchange(input.ReSTIRDIBuffer[i] , nullptr);
            }

            pixelCheck = std::exchange(input.pixelCheck , nullptr);
            albedoImage = std::exchange(input.albedoImage , nullptr);
            roughnessImage = std::exchange(input.roughnessImage , nullptr);
            metallicImage = std::exchange(input.metallicImage , nullptr);
            filteredShadowImage = std::exchange(input.filteredShadowImage , nullptr);
            filteredAmbientOcclusion = std::exchange(input.filteredAmbientOcclusion , nullptr);
            motionVector = std::exchange(input.motionVector , nullptr);
            multiScatter = std::exchange(input.multiScatter , nullptr);
            transmissionLUT = std::exchange(input.transmissionLUT , nullptr);
            skybox = std::exchange(input.skybox , nullptr);

            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            features = std::exchange(input.features , nullptr);
            forwardInfo = std::exchange(input.forwardInfo , nullptr);
            tileLightIndices = std::exchange(input.tileLightIndices , nullptr);
            lightCountPerTile = std::exchange(input.lightCountPerTile , nullptr);
            lightBuffer = std::exchange(input.lightBuffer , nullptr);
            sunBuffer = std::exchange(input.sunBuffer , nullptr);
            skyBuffer = std::exchange(input.skyBuffer , nullptr);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            // the images and the buffers belong to the graph , they are only forgotten
            pipe = nullptr;
            sampler = nullptr;

            for(const auto i : {0 , 1}){
                positionImage[i] = nullptr;
                normalImage[i] = nullptr;
                depthImage[i] = nullptr;
                diffuseImage[i] = nullptr;
                specularImage[i] = nullptr;
                ReSTIRDIBuffer[i] = nullptr;
            }

            pixelCheck = nullptr;
            albedoImage = nullptr;
            roughnessImage = nullptr;
            metallicImage = nullptr;
            filteredShadowImage = nullptr;
            filteredAmbientOcclusion = nullptr;
            motionVector = nullptr;
            multiScatter = nullptr;
            transmissionLUT = nullptr;
            skybox = nullptr;

            generalBuffer = nullptr;
            features = nullptr;
            forwardInfo = nullptr;
            tileLightIndices = nullptr;
            lightCountPerTile = nullptr;
            lightBuffer = nullptr;
            sunBuffer = nullptr;
            skyBuffer = nullptr;

            inputData = {};
            completedInit = false;
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            VulkanContext* context = inputData.context;

            try{

                pipe = std::make_shared<ShaderPipe>(context);

                {
                    Sampler::CreateInfo info{};
                    info.context = context;
                    info.linearClampSampler = true;

                    sampler = std::make_shared<Sampler>(info);
                }

                {
                    std::vector<DescriptorLayoutData> data(26);
                    data[0] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};

                    data[1] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    // 3 .. 14 : the images of the g buffer , of the shadows and the result
                    for(uint32_t i = 3 ; i <= 14 ; i++){
                        data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    }

                    data[15] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[16] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    data[17] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[18] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[19] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    data[20] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[21] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[22] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    // the sun disc
                    data[23] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[24] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[25] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                    pipe->addDescriptorLayout(info);
                }

                {
                    PipelineLayout::CreateInfo info{};
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.context = context;
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {
                    for(auto& id : {0 , 1}){

                        std::vector<DescriptorData2> data(26);

                        data[0] = DescriptorData2::createData(sampler.get() , 0 , 0);
                        data[1] = DescriptorData2::createData(multiScatter , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1 , 0);
                        data[2] = DescriptorData2::createData(skybox , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 2 , 0);

                        data[3] = DescriptorData2::createData(pixelCheck , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);
                        data[4] = DescriptorData2::createData(positionImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 4 , 0);
                        data[5] = DescriptorData2::createData(normalImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 5 , 0);
                        data[6] = DescriptorData2::createData(albedoImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 6 , 0);
                        data[7] = DescriptorData2::createData(depthImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 7 , 0);
                        data[8] = DescriptorData2::createData(roughnessImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0);
                        data[9] = DescriptorData2::createData(metallicImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 9 , 0);
                        data[10] = DescriptorData2::createData(filteredShadowImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 10 , 0);
                        data[11] = DescriptorData2::createData(filteredAmbientOcclusion , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 11 , 0);
                        data[12] = DescriptorData2::createData(diffuseImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 12 , 0);
                        data[13] = DescriptorData2::createData(specularImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 13 , 0);
                        data[14] = DescriptorData2::createData(motionVector , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 14 , 0);

                        data[15] = DescriptorData2::createData(ReSTIRDIBuffer[id] , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 15 , 0);
                        data[16] = DescriptorData2::createData(ReSTIRDIBuffer[!id] , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 16 , 0);

                        data[17] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 17 , 0);
                        data[18] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 18 , 0);
                        data[19] = DescriptorData2::createData(forwardInfo , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 19 , 0);

                        data[20] = DescriptorData2::createData(tileLightIndices , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 20 , 0);
                        data[21] = DescriptorData2::createData(lightCountPerTile , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 21 , 0);
                        data[22] = DescriptorData2::createData(lightBuffer , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 22 , 0);

                        data[23] = DescriptorData2::createData(sunBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 23 , 0);
                        data[24] = DescriptorData2::createData(skyBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 24 , 0);
                        data[25] = DescriptorData2::createData(transmissionLUT , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 25 , 0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = pipe->descriptorLayouts[0].get();

                        pipe->addDescriptorBuffer2(info);
                    }
                }

                {
                    ShaderObject::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = inputData.path;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create LightPass pipe"});
            }

            completedInit = true;
            return {};
        }
    };

    using pLightPass = std::shared_ptr<LightPass>;
}
