#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Sampler/Sampler.hpp"



namespace mox{

    // the indirect light of the screen space : SSGI ( the bounces ) or SSR ( the reflections ) by features.indirectMode , and the light of the sky .
    // shader : indirectShader/indirectCalc.hlsl
    // it runs after the LightPass and before the denoisers : the result is added into diffuseImage ( so it is denoised together with the direct light ) ,
    // viewImage and the hit distance ( .a of diffuse / specular / indirect ) are for the denoisers , indirectImage is the picture of INDIRECT_MODE
    class IndirectLightSum : public GraphPass2{
    public:

        SetTypicalOperators(IndirectLightSum)
        SetTypicalTransferOperators(IndirectLightSum)

        struct CreateInfo{
            VulkanContext* context;
            std::string path;
        };

        explicit IndirectLightSum(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            if(info.path.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create IndirectLightSum , since the path to the shader is empty"});
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~IndirectLightSum(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs{};
            reqs.reserve(24);

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
            image("indirectImage_0" , ImageType::RenderImage2);
            image("indirectImage_1" , ImageType::RenderImage2);
            // the picture of the last frame : the screen space rays take the light from it
            image("HDRImage_0" , ImageType::RenderImage2);
            image("HDRImage_1" , ImageType::RenderImage2);

            image("roughnessImage" , ImageType::RenderImage2);
            image("albedoImage" , ImageType::RenderImage2);
            image("metallicImage" , ImageType::RenderImage2);
            image("viewImage" , ImageType::RenderImage2);
            image("pixelCheck" , ImageType::RenderImage2);
            image("multiScatter" , ImageType::RenderImage2);
            image("skybox" , ImageType::Skybox2);

            buffer("generalBuffer");
            buffer("generalFeatures");

            // Forward+ : the lamps of the tile of the pixel ( the lists are written by the ForwardPlus pass )
            buffer("tileLightIndices");
            buffer("lightBuffer");
            buffer("lightCountPerTile");
            buffer("forwardInfo");

            buffer("sunBuffer");

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [IndirectLightSum]"});

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
                GraphSlot<RenderImage2>{"indirectImage_0" , &indirectImage[0]} , GraphSlot<RenderImage2>{"indirectImage_1" , &indirectImage[1]} ,
                GraphSlot<RenderImage2>{"HDRImage_0" , &HDRImage[0]} , GraphSlot<RenderImage2>{"HDRImage_1" , &HDRImage[1]} ,
                GraphSlot<RenderImage2>{"roughnessImage" , &roughnessImage} , GraphSlot<RenderImage2>{"albedoImage" , &albedoImage} ,
                GraphSlot<RenderImage2>{"metallicImage" , &metallicImage} , GraphSlot<RenderImage2>{"viewImage" , &viewImage} ,
                GraphSlot<RenderImage2>{"pixelCheck" , &pixelCheck} , GraphSlot<RenderImage2>{"multiScatter" , &multiScatter} ,
                GraphSlot<Skybox2>{"skybox" , &skybox} ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer} , GraphSlot<BufferBase2>{"generalFeatures" , &features} ,
                GraphSlot<BufferBase2>{"tileLightIndices" , &tileLightIndices} , GraphSlot<BufferBase2>{"lightBuffer" , &lightBuffer} ,
                GraphSlot<BufferBase2>{"lightCountPerTile" , &lightCountPerTile} , GraphSlot<BufferBase2>{"forwardInfo" , &forwardInfo} ,
                GraphSlot<BufferBase2>{"sunBuffer" , &sunBuffer});
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [IndirectLightSum]"});

            const uint8_t id = inputData.context->sc->frameIndex % 2;
            const uint8_t prev = 1 - id;

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 write = VK_ACCESS_2_SHADER_WRITE_BIT;
            constexpr VkAccessFlags2 uniformRead = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            std::array<VkImageMemoryBarrier2 , 14> imageBar;

            // the g buffer of this frame , the picture of the last frame and the sky : only read
            imageBar[0] = positionImage[id]->translateImage(read , stage);
            imageBar[1] = normalImage[id]->translateImage(read , stage);
            imageBar[2] = depthImage[id]->translateImage(read , stage);
            imageBar[3] = roughnessImage->translateImage(read , stage);
            imageBar[4] = albedoImage->translateImage(read , stage);
            imageBar[5] = metallicImage->translateImage(read , stage);
            imageBar[6] = HDRImage[prev]->translateImage(read , stage);
            imageBar[7] = skybox->translateImage(read , stage);

            // the result : the shader writes the .a ( the hit distance ) and adds the light into diffuse , so these are read and written
            imageBar[8] = viewImage->translateImage(write , stage);
            imageBar[9] = diffuseImage[id]->translateImage(read | write , stage);
            imageBar[10] = specularImage[id]->translateImage(read | write , stage);
            imageBar[11] = indirectImage[id]->translateImage(read | write , stage);

            // only read ( for now )
            imageBar[12] = pixelCheck->translateImage(read , stage);

            // the LUT of the multiple scattering : only read
            imageBar[13] = multiScatter->translateImage(read , stage);

            std::array<VkBufferMemoryBarrier2 , 7> bufBar;

            bufBar[0] = generalBuffer->translateBuffer(uniformRead , stage);
            bufBar[1] = features->translateBuffer(uniformRead , stage);

            // the lamps of the tiles : only read
            bufBar[2] = tileLightIndices->translateBuffer(read , stage);
            bufBar[3] = lightBuffer->translateBuffer(read , stage);
            bufBar[4] = lightCountPerTile->translateBuffer(read , stage);
            bufBar[5] = forwardInfo->translateBuffer(uniformRead , stage);
            bufBar[6] = sunBuffer->translateBuffer(uniformRead , stage);

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

        pShaderPipeType pipe{nullptr};

        CreateInfo inputData{};

        // typicalSampler of the shader : the sky cube is filtered
        pSampler sampler{nullptr};

        RenderImage2* positionImage[2]{nullptr , nullptr};
        RenderImage2* normalImage[2]{nullptr , nullptr};
        RenderImage2* depthImage[2]{nullptr , nullptr};
        RenderImage2* diffuseImage[2]{nullptr , nullptr};
        RenderImage2* specularImage[2]{nullptr , nullptr};
        RenderImage2* indirectImage[2]{nullptr , nullptr};
        RenderImage2* HDRImage[2]{nullptr , nullptr};
        RenderImage2* roughnessImage{nullptr};
        RenderImage2* albedoImage{nullptr};
        RenderImage2* metallicImage{nullptr};
        RenderImage2* viewImage{nullptr};
        RenderImage2* pixelCheck{nullptr};
        RenderImage2* multiScatter{nullptr};
        Skybox2* skybox{nullptr};

        BufferBase2* generalBuffer{nullptr};
        BufferBase2* features{nullptr};
        BufferBase2* tileLightIndices{nullptr};
        BufferBase2* lightBuffer{nullptr};
        BufferBase2* lightCountPerTile{nullptr};
        BufferBase2* forwardInfo{nullptr};
        BufferBase2* sunBuffer{nullptr};

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(IndirectLightSum&& input) noexcept{
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
                indirectImage[i] = std::exchange(input.indirectImage[i] , nullptr);
                HDRImage[i] = std::exchange(input.HDRImage[i] , nullptr);
            }

            roughnessImage = std::exchange(input.roughnessImage , nullptr);
            albedoImage = std::exchange(input.albedoImage , nullptr);
            metallicImage = std::exchange(input.metallicImage , nullptr);
            viewImage = std::exchange(input.viewImage , nullptr);
            pixelCheck = std::exchange(input.pixelCheck , nullptr);
            multiScatter = std::exchange(input.multiScatter , nullptr);
            skybox = std::exchange(input.skybox , nullptr);

            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            features = std::exchange(input.features , nullptr);
            tileLightIndices = std::exchange(input.tileLightIndices , nullptr);
            lightBuffer = std::exchange(input.lightBuffer , nullptr);
            lightCountPerTile = std::exchange(input.lightCountPerTile , nullptr);
            forwardInfo = std::exchange(input.forwardInfo , nullptr);
            sunBuffer = std::exchange(input.sunBuffer , nullptr);

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
                indirectImage[i] = nullptr;
                HDRImage[i] = nullptr;
            }

            roughnessImage = nullptr;
            albedoImage = nullptr;
            metallicImage = nullptr;
            viewImage = nullptr;
            pixelCheck = nullptr;
            multiScatter = nullptr;
            skybox = nullptr;

            generalBuffer = nullptr;
            features = nullptr;
            tileLightIndices = nullptr;
            lightBuffer = nullptr;
            lightCountPerTile = nullptr;
            forwardInfo = nullptr;
            sunBuffer = nullptr;

            inputData = {};
            completedInit = false;
        }

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

                {
                    std::vector<DescriptorLayoutData> data(22);

                    // 0 .. 10 : the images of the g buffer and the result
                    for(uint32_t i = 0 ; i <= 10 ; i++){
                        data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    }

                    // the sky cube and its sampler
                    data[11] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[12] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};

                    data[13] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[14] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    // pixelCheck
                    data[15] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    // Forward+ : the lamps of the tiles
                    data[16] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};   // tileLightIndices
                    data[17] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};   // lightBuffer
                    data[18] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};   // lightCountPerTile
                    data[19] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};   // forwardInfo

                    // the multiple scattering LUT ( the sampler of the sky cube , data[12] , reads it too )
                    data[20] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    // the sun
                    data[21] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

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

                        std::vector<DescriptorData2> data(22);

                        data[0] = DescriptorData2::createData(positionImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0 , 0);
                        data[1] = DescriptorData2::createData(normalImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0);
                        data[2] = DescriptorData2::createData(depthImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0);
                        data[3] = DescriptorData2::createData(roughnessImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);
                        data[4] = DescriptorData2::createData(viewImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 4 , 0);
                        data[5] = DescriptorData2::createData(diffuseImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 5 , 0);
                        data[6] = DescriptorData2::createData(specularImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 6 , 0);
                        data[7] = DescriptorData2::createData(indirectImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 7 , 0);
                        // prevHDRImage : the picture of the previous frame
                        data[8] = DescriptorData2::createData(HDRImage[!id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0);
                        data[9] = DescriptorData2::createData(albedoImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 9 , 0);
                        data[10] = DescriptorData2::createData(metallicImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 10 , 0);

                        data[11] = DescriptorData2::createData(skybox , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 11 , 0);
                        data[12] = DescriptorData2::createData(sampler.get() , 12 , 0);

                        data[13] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 13 , 0);
                        data[14] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 14 , 0);

                        data[15] = DescriptorData2::createData(pixelCheck , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 15 , 0);

                        data[16] = DescriptorData2::createData(tileLightIndices , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 16 , 0);
                        data[17] = DescriptorData2::createData(lightBuffer , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 17 , 0);
                        data[18] = DescriptorData2::createData(lightCountPerTile , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 18 , 0);
                        data[19] = DescriptorData2::createData(forwardInfo , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 19 , 0);

                        data[20] = DescriptorData2::createData(multiScatter , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 20 , 0);

                        data[21] = DescriptorData2::createData(sunBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 21 , 0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = pipe->descriptorLayouts[0].get();

                        pipe->addDescriptorBuffer2(info);
                    }
                }

                {
                    ShaderObjectType::CreateInfo info{};
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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create IndirectLightSum pipe"});
            }

            completedInit = true;
            return {};
        }
    };

    using pIndirectLightSum = std::shared_ptr<IndirectLightSum>;
}
