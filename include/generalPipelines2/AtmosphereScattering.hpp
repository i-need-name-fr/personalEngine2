#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Sampler/Sampler.hpp"


namespace mox{

    class AtmosphereScattering : public GraphPass2{
    public:

        SetTypicalOperators(AtmosphereScattering)
        SetTypicalTransferOperators(AtmosphereScattering)

        struct CreateInfo{
            VulkanContext* context;
            // SkyViewLUT.spv and transitionCube.spv
            std::string skyView , transition;
        };

        explicit AtmosphereScattering(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~AtmosphereScattering(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(6);

            reqs[0].type = moxGraphTypeBuffer;
            reqs[0].name = "skyBuffer";

            reqs[1].type = moxGraphTypeBuffer;
            reqs[1].name = "sunBuffer";

            reqs[2].type = moxGraphTypeImage;
            reqs[2].name = "transmissionLUT";
            reqs[2].imageSpecification.type = ImageType::RenderImage2;

            reqs[3].type = moxGraphTypeImage;
            reqs[3].name = "multiScatterLUT";
            reqs[3].imageSpecification.type = ImageType::RenderImage2;

            reqs[4].type = moxGraphTypeImage;
            reqs[4].name = "SkyView";
            reqs[4].imageSpecification.type = ImageType::RenderImage2;

            reqs[5].type = moxGraphTypeImage;
            reqs[5].name = "skybox";
            reqs[5].imageSpecification.type = ImageType::Skybox2;

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [AtmosphereScattering]"});

            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};

            // the set of both shaders is the same : it is bound once and stays bound when the shader changes
            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.address = pipe->buffers[0]->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 1,  &info);

                uint32_t sets[1] = {0};
                VkDeviceSize offset[] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offset);
            }

            // 1 : the sky view LUT
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            vkCmdDispatch(cmd , (skyView->extent.width + 7) / 8 , (skyView->extent.height + 7) / 8 , 1);

            // the second shader reads what the first one has just written
            {
                const auto barrier = skyView->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

                VkDependencyInfo dependency{};
                dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dependency.bufferMemoryBarrierCount = 0;
                dependency.imageMemoryBarrierCount = 1;
                dependency.pMemoryBarriers = nullptr;
                dependency.pBufferMemoryBarriers = nullptr;
                dependency.pImageMemoryBarriers = &barrier;
                dependency.memoryBarrierCount = 0;
                dependency.pNext = nullptr;
                dependency.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &dependency);
            }

            // 2 : the LUT into the 6 faces of the skybox
            shaders[0] = pipe->shaders[1]->shader;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            vkCmdDispatch(cmd , (skybox->extent.width + 7) / 8 , (skybox->extent.height + 7) / 8 , 6);

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
                GraphSlot<BufferBase2>{"skyBuffer" , &skyBuffer} , GraphSlot<BufferBase2>{"sunBuffer" , &sunBuffer} ,
                GraphSlot<RenderImage2>{"transmissionLUT" , &transmissionLUT} , GraphSlot<RenderImage2>{"multiScatterLUT" , &multiScatterLUT} ,
                GraphSlot<RenderImage2>{"SkyView" , &skyView} , GraphSlot<Skybox2>{"skybox" , &skybox});
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [AtmosphereScattering]"});

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

            std::array<VkBufferMemoryBarrier2 , 2> bufBar;
            bufBar[0] = skyBuffer->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT , stage);
            bufBar[1] = sunBuffer->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT , stage);

            // the LUTs of the transmission and of the multi scattering are only read ( they were computed once ) ,
            // the sky view LUT and the skybox are written in every frame
            std::array<VkImageMemoryBarrier2 , 4> imageBar;
            imageBar[0] = transmissionLUT->translateImage(VK_ACCESS_2_SHADER_READ_BIT , stage);
            imageBar[1] = multiScatterLUT->translateImage(VK_ACCESS_2_SHADER_READ_BIT , stage);
            imageBar[2] = skyView->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , stage);
            imageBar[3] = skybox->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , stage);

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

        // the samplers of the pipe : the sky view LUT is sampled with the clamp , the cube wraps the azimuth
        pSampler clampSampler{nullptr};
        pSampler wrapSampler{nullptr};

        BufferBase2* skyBuffer{nullptr};
        BufferBase2* sunBuffer{nullptr};

        RenderImage2* transmissionLUT{nullptr};
        RenderImage2* multiScatterLUT{nullptr};
        RenderImage2* skyView{nullptr};
        Skybox2* skybox{nullptr};

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(AtmosphereScattering&& input) noexcept{
            if(this == &input) return;
            processDelete();

            pipe = std::move(input.pipe);
            inputData = input.inputData;

            clampSampler = std::move(input.clampSampler);
            wrapSampler = std::move(input.wrapSampler);

            skyBuffer = std::exchange(input.skyBuffer , nullptr);
            sunBuffer = std::exchange(input.sunBuffer , nullptr);
            transmissionLUT = std::exchange(input.transmissionLUT , nullptr);
            multiScatterLUT = std::exchange(input.multiScatterLUT , nullptr);
            skyView = std::exchange(input.skyView , nullptr);
            skybox = std::exchange(input.skybox , nullptr);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            // the buffers and the images belong to the graph , they are only forgotten
            pipe = nullptr;
            clampSampler = nullptr;
            wrapSampler = nullptr;

            skyBuffer = nullptr;
            sunBuffer = nullptr;
            transmissionLUT = nullptr;
            multiScatterLUT = nullptr;
            skyView = nullptr;
            skybox = nullptr;

            inputData = {};
            completedInit = false;
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            VulkanContext* context = inputData.context;

            // the compute shader writes the faces of the cube as the layers of a 2D array , it needs the storage view of the skybox
            if(!skybox->storageView){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create AtmosphereScattering pipe , since the skybox has no storageView ( it needs VK_IMAGE_USAGE_STORAGE_BIT )"});
            }

            try{

                pipe = std::make_shared<ShaderPipeType>(context);

                {
                    Sampler::CreateInfo info{};
                    info.context = context;
                    info.linearClampSampler = true;

                    clampSampler = std::make_shared<Sampler>(info);

                    info.linearClampSampler = false;
                    info.linearWrapUSampler = true;

                    wrapSampler = std::make_shared<Sampler>(info);
                }

                {
                    std::vector<DescriptorLayoutData> data(9);
                    data[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[4] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[5] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[6] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[7] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[8] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};

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
                    std::vector<DescriptorData2> data(9);
                    data[0] = DescriptorData2::createData(skyBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                    data[1] = DescriptorData2::createData(sunBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0);
                    data[2] = DescriptorData2::createData(transmissionLUT , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 2 , 0);
                    data[3] = DescriptorData2::createData(multiScatterLUT , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 3 , 0);
                    data[4] = DescriptorData2::createData(clampSampler.get() , 4 , 0);
                    // the same sky view LUT : written by the first shader ( 5 ) and read by the second one ( 6 )
                    data[5] = DescriptorData2::createData(skyView , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 5 , 0);
                    data[6] = DescriptorData2::createData(skyView , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 6 , 0);
                    // the skybox : the 2D array view of its 6 faces
                    data[7] = DescriptorData2::createData(skybox->storageView , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 7 , 0);
                    data[8] = DescriptorData2::createData(wrapSampler.get() , 8 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->addDescriptorBuffer2(info);
                }

                {
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = inputData.skyView;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};

                    pipe->addShader(info);
                }

                {
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = inputData.transition;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create AtmosphereScattering pipe"});
            }

            completedInit = true;
            return {};
        }
    };

    using pAtmosphereScattering = std::shared_ptr<AtmosphereScattering>;
}
