#pragma once

#include "../ShaderPipe/ShaderPipe.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"


namespace mox{

    class BilateralFiltering : public GraphPass2{
    public:

        SetTypicalOperators(BilateralFiltering)
        SetTypicalTransferOperators(BilateralFiltering)

        struct CreateInfo{

            VulkanContext* context;
            std::string path;

        };

        explicit BilateralFiltering(const CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice , info.context);
                if(!result.has_value()){
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            context = info.context;
            path = info.path;
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(12);
            reqs[0].type = moxGraphTypeImage;
            reqs[0].name = "normalImage_0";
            reqs[0].imageSpecification.type = ImageType::RenderImage2;

            reqs[1].type = moxGraphTypeImage;
            reqs[1].name = "normalImage_1";
            reqs[1].imageSpecification.type = ImageType::RenderImage2;

            reqs[2].type = moxGraphTypeImage;
            reqs[2].name = "depthImage_0";
            reqs[2].imageSpecification.type = ImageType::RenderImage2;

            reqs[3].type = moxGraphTypeImage;
            reqs[3].name = "depthImage_1";
            reqs[3].imageSpecification.type = ImageType::RenderImage2;
            
            reqs[4].type = moxGraphTypeImage;
            reqs[4].name = "filteredShadowImage";
            reqs[4].imageSpecification.type = ImageType::RenderImage2;

            reqs[5].type = moxGraphTypeImage;
            reqs[5].name = "pixelCheck";
            reqs[5].imageSpecification.type = ImageType::RenderImage2;

            reqs[6].type = moxGraphTypeImage;
            reqs[6].name = "shadowImage_0";
            reqs[6].imageSpecification.type = ImageType::RenderImage2;

            reqs[7].type = moxGraphTypeImage;
            reqs[7].name = "shadowImage_1";
            reqs[7].imageSpecification.type = ImageType::RenderImage2;

            reqs[8].type = moxGraphTypeImage;
            reqs[8].name = "ambientOcclusionImage_0";
            reqs[8].imageSpecification.type = ImageType::RenderImage2;

            reqs[9].type = moxGraphTypeImage;
            reqs[9].name = "ambientOcclusionImage_1";
            reqs[9].imageSpecification.type = ImageType::RenderImage2;

            reqs[10].type = moxGraphTypeImage;
            reqs[10].name = "filteredAmbientOcclusion";
            reqs[10].imageSpecification.type = ImageType::RenderImage2;

            reqs[11].type = moxGraphTypeBuffer;
            reqs[11].name = "generalFeatures";
            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [BF]"});

            const uint8_t currentImage = context->sc->frameIndex %2;

            const uint32_t x = (context->sc->extent.width + 7) / 8;
            const uint32_t y = (context->sc->extent.height + 7) / 8;

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            // shadow
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

            uint32_t push{0};

            vkCmdPushConstants(cmd , pipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT, 0 , sizeof(uint32_t) , &push);

            vkCmdDispatch(cmd , x , y, 1);

            // AO
            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.address = pipe->buffers[currentImage + 2]->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 1,  &info);

                uint32_t sets[1] = {0};
                VkDeviceSize offset[] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offset);
            }

            push = 1;

            vkCmdPushConstants(cmd , pipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT, 0 , sizeof(uint32_t) , &push);

            vkCmdDispatch(cmd , x , y, 1);

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
                GraphSlot<RenderImage2>{"depthImage_0" , &depthImage[0]} , GraphSlot<RenderImage2>{"depthImage_1" , &depthImage[1]} ,
                GraphSlot<RenderImage2>{"normalImage_0" , &normalImage[0]} , GraphSlot<RenderImage2>{"normalImage_1" , &normalImage[1]} ,
                GraphSlot<RenderImage2>{"shadowImage_0" , &shadowImage[0]} , GraphSlot<RenderImage2>{"shadowImage_1" , &shadowImage[1]} ,
                GraphSlot<RenderImage2>{"ambientOcclusionImage_0" , &ambientOcclusionImage[0]} , GraphSlot<RenderImage2>{"ambientOcclusionImage_1" , &ambientOcclusionImage[1]} ,
                GraphSlot<RenderImage2>{"filteredAmbientOcclusion" , &filteredAmbientOcclusionImage} ,
                GraphSlot<RenderImage2>{"filteredShadowImage" , &filteredShadowImage} ,
                GraphSlot<RenderImage2>{"pixelCheck" , &pixelCheck} ,
                GraphSlot<BufferBase2>{"generalFeatures" , &features});
            if(!bound.has_value()) return bound;
            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            const uint8_t id = context->sc->frameIndex %2;

            std::array<VkImageMemoryBarrier2 , 7> imageBar;
            imageBar[0] = shadowImage[id]->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[1] = ambientOcclusionImage[id]->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            imageBar[2] = normalImage[id]->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[3] = depthImage[id]->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            imageBar[4] = filteredAmbientOcclusionImage->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[5] = filteredShadowImage->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            imageBar[6] = pixelCheck->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);


            VkBufferMemoryBarrier2 bufBar = features->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 1;
            info.imageMemoryBarrierCount = 7;
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

        VulkanContext* context{nullptr};
        std::string path;

        pShaderPipe pipe = nullptr;

        RenderImage2* depthImage[2]{nullptr , nullptr};
        RenderImage2* normalImage[2]{nullptr , nullptr};
        RenderImage2* shadowImage[2]{nullptr , nullptr};
        RenderImage2* ambientOcclusionImage[2]{nullptr , nullptr};

        RenderImage2* filteredAmbientOcclusionImage{nullptr};
        RenderImage2* filteredShadowImage{nullptr};  
        
        RenderImage2* pixelCheck{nullptr};

        BufferBase2* features{nullptr};

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(BilateralFiltering&& input) noexcept{
            if(this == &input) return;
            processDelete();

            context = std::exchange(input.context , nullptr);
            path = std::move(input.path);
            pipe = std::move(input.pipe);

            for(const auto i : {0 , 1}){
                depthImage[i] = std::exchange(input.depthImage[i] , nullptr);
                normalImage[i] = std::exchange(input.normalImage[i] , nullptr);
                shadowImage[i] = std::exchange(input.shadowImage[i] , nullptr);
                ambientOcclusionImage[i] = std::exchange(input.ambientOcclusionImage[i] , nullptr);
            }

            filteredAmbientOcclusionImage = std::exchange(input.filteredAmbientOcclusionImage , nullptr);
            filteredShadowImage = std::exchange(input.filteredShadowImage , nullptr);
            pixelCheck = std::exchange(input.pixelCheck , nullptr);
            features = std::exchange(input.features , nullptr);

            // the moved from pass is not initialized any more : its pipe is gone
            completedInit = std::exchange(input.completedInit , false);
        }

        void processDelete() noexcept{
            // the pointers belong to the graph , so they are only forgotten here , nothing is destroyed
            pipe = nullptr;
            context = nullptr;
            path.clear();

            for(const auto i : {0 , 1}){
                depthImage[i] = nullptr;
                normalImage[i] = nullptr;
                shadowImage[i] = nullptr;
                ambientOcclusionImage[i] = nullptr;
            }

            filteredAmbientOcclusionImage = nullptr;
            filteredShadowImage = nullptr;
            pixelCheck = nullptr;
            features = nullptr;
            completedInit = false;
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            try{
                pipe = std::make_shared<ShaderPipe>(context);

                {
                    std::vector<DescriptorLayoutData> data(6);
                    data[0] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
 
                    data[5] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                    pipe->addDescriptorLayout(info);
                }

                VkPushConstantRange range{};
                range.offset = 0;
                range.size = sizeof(uint32_t);
                range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

                {
                    PipelineLayout::CreateInfo info{};
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.context = context;
                    info.ranges ={range};
                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {   // shadows
                    for(auto& id : { 0 , 1}){
                        std::vector<DescriptorData2> data(6);
                        data[0] = DescriptorData2::createData(shadowImage[id] , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 0 , 0);
                        data[1] = DescriptorData2::createData(normalImage[id] , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1 , 0);
                        data[2] = DescriptorData2::createData(depthImage[id] , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 2 , 0);

                        data[3] = DescriptorData2::createData(filteredShadowImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);
                        data[4] = DescriptorData2::createData(pixelCheck , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 4 , 0);

                        data[5] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 5 , 0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = pipe->descriptorLayouts[0].get();

                        pipe->addDescriptorBuffer2(info);
                    }
                }

                {   // shadows
                    for(auto& id : { 0 , 1}){
                        std::vector<DescriptorData2> data(6);
                        data[0] = DescriptorData2::createData(ambientOcclusionImage[id] , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 0 , 0);
                        data[1] = DescriptorData2::createData(normalImage[id] , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1 , 0);
                        data[2] = DescriptorData2::createData(depthImage[id] , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 2 , 0);

                        data[3] = DescriptorData2::createData(filteredAmbientOcclusionImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);
                        data[4] = DescriptorData2::createData(pixelCheck , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 4 , 0);

                        data[5] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 5 , 0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = pipe->descriptorLayouts[0].get();

                        pipe->addDescriptorBuffer2(info);
                    }
                }

                {
                    ShaderObject::CreateInfo info{};
                    info.context=  context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = path;
                    info.pushes = {range};
                    info.layouts = {pipe->descriptorLayouts[0].get()};

                    pipe->addShader(info);
                }
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION ,"failed to create BF pipe"});
            }

            completedInit = true;
            return {};
        }
    };
}