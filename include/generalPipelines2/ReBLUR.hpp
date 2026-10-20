#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../Object2/Mesh2.hpp"


namespace mox{

    class ReBLUR : public GraphPass2{
    public:

        SetTypicalOperators(ReBLUR)
        SetTypicalTransferOperators(ReBLUR)

        struct CreateInfo{
            VulkanContext* context;
            std::string spatial , temporal;
        };

        explicit ReBLUR(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice,  info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~ReBLUR(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(13);

            reqs[0].type = moxGraphTypeImage;
            reqs[0].name = "depthImage_0";
            reqs[0].imageSpecification.type = ImageType::RenderImage2;

            reqs[1].type = moxGraphTypeImage;
            reqs[1].name = "depthImage_1";
            reqs[1].imageSpecification.type = ImageType::RenderImage2;

            reqs[2].type = moxGraphTypeImage;
            reqs[2].name = "normalImage_0";
            reqs[2].imageSpecification.type = ImageType::RenderImage2;

            reqs[3].type = moxGraphTypeImage;
            reqs[3].name = "normalImage_1";
            reqs[3].imageSpecification.type = ImageType::RenderImage2;

            reqs[4].type = moxGraphTypeImage;
            reqs[4].name = "positionImage_0";
            reqs[4].imageSpecification.type = ImageType::RenderImage2;

            reqs[5].type = moxGraphTypeImage;
            reqs[5].name = "positionImage_1";
            reqs[5].imageSpecification.type = ImageType::RenderImage2;

            reqs[6].type = moxGraphTypeImage;
            reqs[6].name = "diffuseImage_0";
            reqs[6].imageSpecification.type = ImageType::RenderImage2;

            reqs[7].type = moxGraphTypeImage;
            reqs[7].name = "diffuseImage_1";
            reqs[7].imageSpecification.type = ImageType::RenderImage2;

            reqs[8].type = moxGraphTypeImage;
            reqs[8].name = "indirectImage_0";
            reqs[8].imageSpecification.type = ImageType::RenderImage2;

            reqs[9].type = moxGraphTypeImage;
            reqs[9].name = "indirectImage_1";
            reqs[9].imageSpecification.type = ImageType::RenderImage2;

            reqs[10].type = moxGraphTypeImage;
            reqs[10].name = "pixelCheck";
            reqs[10].imageSpecification.type = ImageType::RenderImage2;

            reqs[11].type = moxGraphTypeImage;
            reqs[11].name = "motionVector";
            reqs[11].imageSpecification.type = ImageType::RenderImage2;

            reqs[12].type = moxGraphTypeBuffer;
            reqs[12].name = "generalFeatures";

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [ReBLUR]"});

            executeShader(cmd , 0);
            executeShader(cmd , 1);

            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{

            return true;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindSources(sources ,
                GraphSlot<RenderImage2>{"depthImage_0" , &depthImage[0]} , GraphSlot<RenderImage2>{"depthImage_1" , &depthImage[1]} ,
                GraphSlot<RenderImage2>{"normalImage_0" , &normalImage[0]} , GraphSlot<RenderImage2>{"normalImage_1" , &normalImage[1]} ,
                GraphSlot<RenderImage2>{"positionImage_0" , &positionImage[0]} , GraphSlot<RenderImage2>{"positionImage_1" , &positionImage[1]} ,
                GraphSlot<RenderImage2>{"diffuseImage_0" , &diffuseImage[0]} , GraphSlot<RenderImage2>{"diffuseImage_1" , &diffuseImage[1]} ,
                GraphSlot<RenderImage2>{"indirectImage_0" , &indirectImage[0]} , GraphSlot<RenderImage2>{"indirectImage_1" , &indirectImage[1]} ,
                GraphSlot<RenderImage2>{"pixelCheck" , &pixelCheck} , GraphSlot<RenderImage2>{"motionVector" , &motionVector} ,
                GraphSlot<BufferBase2>{"generalFeatures" , &features});
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [ReBLUR]"});

            const uint8_t id = inputData.context->sc->frameIndex % 2;
            const uint8_t prev = 1 - id;

            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 readWrite = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

            std::array<VkImageMemoryBarrier2 , 16> imageBar;

            imageBar[0] = depthImage[id]->translateImage(read , stage);
            imageBar[1] = depthImage[prev]->translateImage(read , stage);
            imageBar[2] = normalImage[id]->translateImage(read , stage);
            imageBar[3] = normalImage[prev]->translateImage(read , stage);
            imageBar[4] = positionImage[id]->translateImage(read , stage);
            imageBar[5] = positionImage[prev]->translateImage(read , stage);

            imageBar[6] = diffuseImage[id]->translateImage(readWrite , stage);
            imageBar[7] = diffuseImage[prev]->translateImage(read , stage);
            imageBar[8] = diffuseCount[id]->translateImage(readWrite , stage);
            imageBar[9] = diffuseCount[prev]->translateImage(read , stage);

            imageBar[10] = indirectImage[id]->translateImage(readWrite , stage);
            imageBar[11] = indirectImage[prev]->translateImage(read , stage);
            imageBar[12] = indirectCount[id]->translateImage(readWrite , stage);
            imageBar[13] = indirectCount[prev]->translateImage(read , stage);

            imageBar[14] = motionVector->translateImage(read , stage);
            imageBar[15] = pixelCheck->translateImage(read , stage);

            VkBufferMemoryBarrier2 bufBar = features->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT , stage);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 1;
            info.imageMemoryBarrierCount = static_cast<uint32_t>(imageBar.size());
            info.pMemoryBarriers = nullptr;
            info.pBufferMemoryBarriers = &bufBar;
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

        pRenderImage2 diffuseCount[2]{nullptr};
        pRenderImage2 indirectCount[2]{nullptr};
        pRenderImage2 outputImage{nullptr};

        RenderImage2* positionImage[2]{nullptr , nullptr};
        RenderImage2* depthImage[2]{nullptr , nullptr};
        RenderImage2* normalImage[2]{nullptr , nullptr};
        RenderImage2* indirectImage[2]{nullptr , nullptr};
        RenderImage2* diffuseImage[2]{nullptr , nullptr};
        // 10
        RenderImage2* pixelCheck{nullptr};
        RenderImage2* motionVector{nullptr};
        // 12
        BufferBase2* features{nullptr};
        // 13
        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(ReBLUR&& input) noexcept{
            if(this == &input) return;
            processDelete();

            pipe = std::move(input.pipe);
            inputData = input.inputData;

            // the images of the pass
            for(const auto i : {0 , 1}){
                diffuseCount[i] = std::move(input.diffuseCount[i]);
                indirectCount[i] = std::move(input.indirectCount[i]);
            }
            outputImage = std::move(input.outputImage);

            // the images and the buffer of the graph
            for(const auto i : {0 , 1}){
                positionImage[i] = std::exchange(input.positionImage[i] , nullptr);
                depthImage[i] = std::exchange(input.depthImage[i] , nullptr);
                normalImage[i] = std::exchange(input.normalImage[i] , nullptr);
                indirectImage[i] = std::exchange(input.indirectImage[i] , nullptr);
                diffuseImage[i] = std::exchange(input.diffuseImage[i] , nullptr);
            }
            pixelCheck = std::exchange(input.pixelCheck , nullptr);
            motionVector = std::exchange(input.motionVector , nullptr);
            features = std::exchange(input.features , nullptr);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            // the pipe and the images of the pass are the ones of the pass , the pointers of the graph are only forgotten
            pipe = nullptr;

            for(const auto i : {0 , 1}){
                diffuseCount[i] = nullptr;
                indirectCount[i] = nullptr;
            }
            outputImage = nullptr;

            for(const auto i : {0 , 1}){
                positionImage[i] = nullptr;
                depthImage[i] = nullptr;
                normalImage[i] = nullptr;
                indirectImage[i] = nullptr;
                diffuseImage[i] = nullptr;
            }
            pixelCheck = nullptr;
            motionVector = nullptr;
            features = nullptr;

            inputData = {};
            completedInit = false;
        }

        void executeShader(VkCommandBuffer& cmd , uint32_t currentShader) noexcept{
            const uint8_t currentImage = inputData.context->sc->frameIndex %2;

            const uint32_t x = (inputData.context->sc->extent.width + 7) / 8;
            const uint32_t y = (inputData.context->sc->extent.height + 7) / 8;

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            // write shader
            {   
                auto barrier = outputImage->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 1;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = &barrier;
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);
            }

            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.address = pipe->buffers[currentImage + currentShader * 2]->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 1,  &info);

                uint32_t sets[1] = {0};
                VkDeviceSize offset[] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offset);
            }

            vkCmdDispatch(cmd , x , y, 1);

            // read
            {   
                auto barrier = outputImage->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 1;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = &barrier;
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);
            }

            // the spatial shader reads the frame count that the temporal one has just written : the memory of that image
            // needs its own barrier ( the barrier above is only for the output image )
            {
                RenderImage2* count = currentShader == 0 ? diffuseCount[currentImage].get() : indirectCount[currentImage].get();
                auto barrier = count->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 1;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = &barrier;
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);
            }

            shaders[0] = pipe->shaders[1]->shader;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.address = pipe->buffers[currentImage + currentShader * 2]->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 1,  &info);

                uint32_t sets[1] = {0};
                VkDeviceSize offset[] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offset);
            }

            vkCmdDispatch(cmd , x , y, 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe(){
            VulkanContext* context = inputData.context;
            try{
                pipe = std::make_shared<ShaderPipeType>(context);
                {
                    // creating the images
                    RenderImage2::CreateInfo info{};
                    info.context = context;
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.clear.float32[0] = 0.0f;
                    info.clear.float32[1] = 0.0f;
                    info.clear.float32[2] = 0.0f;
                    info.clear.float32[3] = 0.0f;
                    info.layerCount = 1;
                    info.mipMapCount = 1;
                    info.extent = context->sc->extent;
                    info.imageFormat = VK_FORMAT_R32_SFLOAT;

                    diffuseCount[0] = std::make_shared<RenderImage2>(info);
                    indirectCount[0] = std::make_shared<RenderImage2>(info);

                    diffuseCount[1] = std::make_shared<RenderImage2>(info);
                    indirectCount[1] = std::make_shared<RenderImage2>(info);

                    info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

                    outputImage = std::make_shared<RenderImage2>(info);
                }

                {
                    std::vector<DescriptorLayoutData> data(14);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    data[4] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[5] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[6] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[7] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    data[8] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    data[9] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[10] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    data[11] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[12] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    data[13] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

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

                        std::vector<DescriptorData2> data(14);

                        data[0] = DescriptorData2::createData(diffuseImage[id] ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0 , 0 );
                        data[1] = DescriptorData2::createData(diffuseImage[!id] ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0 );
                        data[2] = DescriptorData2::createData(diffuseCount[id].get() ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0 );
                        data[3] = DescriptorData2::createData(diffuseCount[!id].get() ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0 );

                        data[4] = DescriptorData2::createData(depthImage[id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 4 , 0 );
                        data[5] = DescriptorData2::createData(depthImage[!id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 5 , 0 );
                        data[6] = DescriptorData2::createData(normalImage[id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 6 , 0 );
                        data[7] = DescriptorData2::createData(normalImage[!id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 7 , 0 );

                        data[8] = DescriptorData2::createData(motionVector ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0 );

                        data[9] = DescriptorData2::createData(positionImage[id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 9 , 0 );
                        data[10] = DescriptorData2::createData(positionImage[!id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 10 , 0 );

                        data[11] = DescriptorData2::createData(outputImage.get() ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 11 , 0 );
                        data[12] = DescriptorData2::createData(pixelCheck ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 12 , 0 );

                        data[13] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 13 , 0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = pipe->descriptorLayouts[0].get();

                        pipe->addDescriptorBuffer2(info);
                    }

                    for(auto& id : {0 , 1}){

                        std::vector<DescriptorData2> data(14);

                        data[0] = DescriptorData2::createData(indirectImage[id] ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0 , 0 );
                        data[1] = DescriptorData2::createData(indirectImage[!id] ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0 );
                        data[2] = DescriptorData2::createData(indirectCount[id].get() ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0 );
                        data[3] = DescriptorData2::createData(indirectCount[!id].get() ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0 );

                        data[4] = DescriptorData2::createData(depthImage[id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 4 , 0 );
                        data[5] = DescriptorData2::createData(depthImage[!id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 5 , 0 );
                        data[6] = DescriptorData2::createData(normalImage[id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 6 , 0 );
                        data[7] = DescriptorData2::createData(normalImage[!id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 7 , 0 );

                        data[8] = DescriptorData2::createData(motionVector ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0 );

                        data[9] = DescriptorData2::createData(positionImage[id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 9 , 0 );
                        data[10] = DescriptorData2::createData(positionImage[!id] ,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 10 , 0 );

                        data[11] = DescriptorData2::createData(outputImage.get() ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 11 , 0 );
                        data[12] = DescriptorData2::createData(pixelCheck ,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 12 , 0 );

                        data[13] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 13 , 0);

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
                    info.nextStage=0;
                    info.filePath = inputData.temporal;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};
                    pipe->addShader(info);
                }

                {
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage=0;
                    info.filePath = inputData.spatial;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};
                    pipe->addShader(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create ReBLUR pipe"});
            }
            completedInit = true;
            return {};
        }
    };

    using pReBLUR = std::shared_ptr<ReBLUR>;
}