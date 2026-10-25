#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../Object2/Mesh2.hpp"

namespace mox{

    class DeferredRender : public GraphPass2{
    public: 

        DeferredRender() = delete;
        DeferredRender(DeferredRender& input) = delete;
        DeferredRender(const DeferredRender& input) = delete;
        DeferredRender& operator=(DeferredRender& input) = delete;
        DeferredRender& operator=(const DeferredRender& input) = delete;

        DeferredRender(DeferredRender&& input){
            processCopy(std::move(input));
        }

        DeferredRender& operator=(DeferredRender&& input){
            processCopy(std::move(input));
            return *this;
        }


        struct CreateInfo{
            VulkanContext* context;
            TextureManager2* manager;

            std::string vertex , fragment;
        };

        DeferredRender(const CreateInfo &info){
            if(!info.manager){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Deferred Render , since the Texture manager2 is nullptr"});
                THROW_MESSAGE;
            }

            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result.error());
                THROW_MESSAGE;
            }
            context = info.context;
            manager = info.manager;
            inputData = info;
        }

        ~DeferredRender(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(12);

            reqs[0].type = moxGraphTypeImage;
            reqs[0].name = "positionImage_0";
            reqs[0].imageSpecification.type = ImageType::RenderImage2;

            reqs[1].type = moxGraphTypeImage;
            reqs[1].name = "positionImage_1";
            reqs[1].imageSpecification.type = ImageType::RenderImage2;

            reqs[2].type = moxGraphTypeImage;
            reqs[2].name = "normalImage_0";
            reqs[2].imageSpecification.type = ImageType::RenderImage2;

            reqs[3].type = moxGraphTypeImage;
            reqs[3].name = "normalImage_1";
            reqs[3].imageSpecification.type = ImageType::RenderImage2;

            reqs[4].type = moxGraphTypeImage;
            reqs[4].name = "depthImage_0";
            reqs[4].imageSpecification.type = ImageType::RenderImage2;

            reqs[5].type = moxGraphTypeImage;
            reqs[5].name = "depthImage_1";
            reqs[5].imageSpecification.type = ImageType::RenderImage2;
            
            reqs[6].type = moxGraphTypeImage;
            reqs[6].name = "albedoImage";
            reqs[6].imageSpecification.type = ImageType::RenderImage2;

            reqs[7].type = moxGraphTypeImage;
            reqs[7].name = "pixelCheck";
            reqs[7].imageSpecification.type = ImageType::RenderImage2;

            reqs[8].type = moxGraphTypeImage;
            reqs[8].name = "metallicImage";
            reqs[8].imageSpecification.type = ImageType::RenderImage2;

            reqs[9].type = moxGraphTypeImage;
            reqs[9].name = "roughnessImage";
            reqs[9].imageSpecification.type = ImageType::RenderImage2;

            reqs[10].type = moxGraphTypeBuffer;
            reqs[10].name = "generalBuffer";

            reqs[11].type = moxGraphTypeImage;
            reqs[11].name = "objectAmbientImage";
            reqs[11].imageSpecification.type = ImageType::RenderImage2;

            return reqs;
         } 

        [[nodiscard]] [[gnu::hot]]  const bool requestExecution() const noexcept override final{
            // since its general type pipeline
            return true;
        } 

        [[nodiscard]]  const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindSources(sources ,
                GraphSlot<RenderImage2>{"positionImage_0" , &positionImage[0]} , GraphSlot<RenderImage2>{"positionImage_1" , &positionImage[1]} ,
                GraphSlot<RenderImage2>{"normalImage_0" , &normalImage[0]} , GraphSlot<RenderImage2>{"normalImage_1" , &normalImage[1]} ,
                GraphSlot<RenderImage2>{"depthImage_0" , &depthImage[0]} , GraphSlot<RenderImage2>{"depthImage_1" , &depthImage[1]} ,
                GraphSlot<RenderImage2>{"albedoImage" , &albedoImage} , GraphSlot<RenderImage2>{"pixelCheck" , &pixelCheck} ,
                GraphSlot<RenderImage2>{"metallicImage" , &metallicImage} , GraphSlot<RenderImage2>{"roughnessImage" , &roughnessImage} ,
                GraphSlot<RenderImage2>{"objectAmbientImage" , &objectAmbientImage} ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer});
            if(!bound.has_value()) return bound;

            return processInit(inputData);
        } 

        [[nodiscard]] [[gnu::hot]]  const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translateBarriers , since the pipeline wasnt created [DeferredRender]"});
            const uint8_t currentImage = context->sc->frameIndex %2;

            // the g buffer is written through the color attachments of the dynamic rendering : the load op clears them ,
            // so the old contents are not needed , and the images stay in GENERAL ( the next passes read them as storage images )
            std::array<VkImageMemoryBarrier2 , 8> barriers{};
            barriers[0] = albedoImage->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[1] = positionImage[currentImage]->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[2] = normalImage[currentImage]->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[3] = depthImage[currentImage]->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[4] = roughnessImage->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[5] = metallicImage->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[6] = pixelCheck->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[7] = objectAmbientImage->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            VkBufferMemoryBarrier2 buffer{};
            buffer = generalBuffer->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 1;
            info.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            info.pMemoryBarriers = nullptr;
            info.pBufferMemoryBarriers = &buffer;
            info.pImageMemoryBarriers = barriers.data();
            info.memoryBarrierCount = 0;
            info.pNext = nullptr;
            info.dependencyFlags = 0;

            vkCmdPipelineBarrier2(cmd , &info);

            return {};
        } 

        [[nodiscard]] [[gnu::hot]]  ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to executePass , since the pipeline wasnt created [DeferredRender]"});

            VkShaderEXT shaders[2] = {pipe->shaders[0]->shader  , pipe->shaders[1]->shader };
            VkShaderStageFlagBits stages[2] = {VK_SHADER_STAGE_VERTEX_BIT , VK_SHADER_STAGE_FRAGMENT_BIT};
            const uint8_t currentImage = context->sc->frameIndex %2;

            cmdBindShadersEXT(cmd , 2 , stages , shaders);

            // main setup for pipeline
            VkViewport viewport{};
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            viewport.x = 0.0f;
            viewport.width = static_cast<float>(inputData.context->sc->extent.width);
            // The ONE Y flip for Vulkan clip space lives here , as a negative height.
            // Doing it in the projection instead would break every shader that
            // rebuilds a ray through invViewProj.
            viewport.y = static_cast<float>(inputData.context->sc->extent.height);
            viewport.height = -static_cast<float>(inputData.context->sc->extent.height);

            VkRect2D scissor{};
            scissor.extent = {inputData.context->sc->extent.width , inputData.context->sc->extent.height};
            scissor.offset = {0 , 0};

            vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
            // Upright viewport + the single Y flip in projOnly leaves the cube's outward
            // faces counter clockwise at the rasteriser. The earlier setup flipped Y
            // twice ( projection AND a negative viewport height ) and every face came
            // out back facing , so the whole scene was culled and only sky remained.
            vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
            vkCmdSetDepthTestEnable(cmd, VK_TRUE);
            vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
            vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_LESS);
            vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            vkCmdSetViewportWithCount(cmd, 1, &viewport);
            vkCmdSetScissorWithCount(cmd, 1, &scissor);

            static std::array<VkVertexInputBindingDescription2EXT, 1> vertexBindings;

            vertexBindings[0].sType     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;
            vertexBindings[0].pNext     = nullptr;
            vertexBindings[0].binding   = 0;
            vertexBindings[0].stride    = sizeof(Vertex);
            vertexBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vertexBindings[0].divisor   = 1;

            static std::array<VkVertexInputAttributeDescription2EXT , 5> vertexAttributes;

            // === binding 0 - per-vertex ===
            vertexAttributes[0] = { .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .pNext = nullptr, .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex , position) }; // position
            vertexAttributes[1] = { .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .pNext = nullptr, .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex , normal) }; // normal
            vertexAttributes[2] = { .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .pNext = nullptr, .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex , tangent) }; // tangent
            vertexAttributes[3] = { .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .pNext = nullptr, .location = 3, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex , bitangent) }; // bitangent
            vertexAttributes[4] = { .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .pNext = nullptr, .location = 4, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    // UV is NOT at 64 : alignas(16) on the vec3 members does not push a vec2 to the next 16 bytes
              .offset = offsetof(Vertex , UV) }; // uv

            vkCmdSetRasterizerDiscardEnable(cmd , VK_FALSE);
            vkCmdSetDepthBiasEnable(cmd , VK_FALSE);
            vkCmdSetStencilTestEnable(cmd , VK_FALSE);
            vkCmdSetPrimitiveRestartEnable(cmd , VK_FALSE);
            cmdSetPolygonModeEXT(cmd , VK_POLYGON_MODE_FILL);
            cmdSetRasterizationSamplesEXT(cmd , VK_SAMPLE_COUNT_1_BIT);
            const VkSampleMask sampleMask = 0xFFFFFFFF;
            cmdSetSampleMaskEXT(cmd , VK_SAMPLE_COUNT_1_BIT , &sampleMask);
            cmdSetAlphaToCoverageEnableEXT(cmd , VK_FALSE);

            // the eight color attachments : no blending , everything is written
            {
                const VkBool32 blend[8] = {VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE};
                const VkColorComponentFlags mask[8] = {
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT ,
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT ,
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT ,
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT ,
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT ,
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT ,
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT ,
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};

                cmdSetColorBlendEnableEXT(cmd , 0 , 8 , blend);
                cmdSetColorWriteMaskEXT(cmd , 0 , 8 , mask);
            }

            // the vertex shader writes SV_ShadingRate , so the primitive rate replaces the pipeline one
            const VkExtent2D shadingRate = {1 , 1};
            const VkFragmentShadingRateCombinerOpKHR combiners[2] = {VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR , VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR};
            cmdSetFragmentShadingRateKHR(cmd , &shadingRate , combiners);

            cmdSetVertexInputEXT(cmd , static_cast<uint32_t>(vertexBindings.size()) , vertexBindings.data() , static_cast<uint32_t>(vertexAttributes.size()) , vertexAttributes.data());
            // vertices / indices
            VkDeviceSize offset = 0;

            vkCmdBindVertexBuffers(cmd , 0 , 1, &Mesh2::vertexBuffer->buffer , &offset);
            vkCmdBindIndexBuffer(cmd , Mesh2::indicesBuffer->buffer , offset , VK_INDEX_TYPE_UINT32);

            // descriptor buffers
            {
                std::array<VkDescriptorBufferBindingInfoEXT, 2> info;
                info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[0].address = pipe->buffers[0]->pBuffer;
                info[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[0].pNext = nullptr;

                info[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[1].address = inputData.manager->dbuffer->pBuffer;
                info[1].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[1].pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 2,  info.data());

                uint32_t sets[2] = {0 , 1};
                VkDeviceSize offset[2] = {0 , 0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_GRAPHICS , pipe->layout->layout , 0 , 2 , sets , offset);
            }

            {
                VkImageMemoryBarrier2 barrier = depthBuffer->translateImage(VK_ACCESS_2_TRANSFER_WRITE_BIT ,VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT );
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.pNext = nullptr;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd , &dep);

                depthBuffer->cleanTheImage(cmd);

                barrier = depthBuffer->translateImage(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT ,VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT );
                vkCmdPipelineBarrier2(cmd , &dep);
                
            }

            // dynamic rendering setup


            {
                VkRenderingAttachmentInfo depthInfo{};
                depthInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                depthInfo.pNext = nullptr;
                depthInfo.imageView = depthBuffer->view;
                depthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depthInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depthInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                depthInfo.clearValue.depthStencil = {1.0f , 0};
                depthInfo.resolveMode = VK_RESOLVE_MODE_NONE;
                depthInfo.resolveImageView = nullptr;
                depthInfo.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                // the g buffer : the order is the order of SV_Target in the fragment shader
                RenderImage2* targets[8] = {albedoImage , positionImage[currentImage] , normalImage[currentImage] , depthImage[currentImage] , roughnessImage , metallicImage , pixelCheck , objectAmbientImage};
                std::array<VkRenderingAttachmentInfo , 8> colorInfo{};
                for(uint32_t i = 0 ; i < 8 ; i++){
                    colorInfo[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    colorInfo[i].pNext = nullptr;
                    colorInfo[i].imageView = targets[i]->view;
                    colorInfo[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    colorInfo[i].resolveMode = VK_RESOLVE_MODE_NONE;
                    colorInfo[i].resolveImageView = nullptr;
                    colorInfo[i].resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    // the pixels that no triangle covers have to be empty ( 0 , and pixelCheck = UNDEFINED_PIXEL )
                    colorInfo[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    colorInfo[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    colorInfo[i].clearValue.color = {};
                }

                VkRenderingInfo info{};
                info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                info.colorAttachmentCount = static_cast<uint32_t>(colorInfo.size());
                info.pColorAttachments = colorInfo.data();
                info.pDepthAttachment = &depthInfo;
                info.flags = 0;
                info.layerCount = 1;
                info.renderArea.extent = {inputData.context->sc->extent.width , inputData.context->sc->extent.height};
                info.renderArea.offset = {0 ,0};
                info.pStencilAttachment = nullptr;
                vkCmdBeginRendering(cmd , &info);
            }


            // draw
            vkCmdDrawIndexedIndirect(cmd , Mesh2::indirectCmdBuffer->buffer , 0 , Mesh2::totalCmdCount , sizeof(VkDrawIndexedIndirectCommand));

            vkCmdEndRendering(cmd);
            shaders[0] = {nullptr};
            shaders[1] = {nullptr};
            cmdBindShadersEXT(cmd , 2 , stages , shaders);

            return {};
        } 

        pRenderImage2 depthBuffer{nullptr};

    private:

        pShaderPipeType pipe{nullptr};

        VulkanContext* context{nullptr};
        TextureManager2* manager{nullptr};
        
        RenderImage2* positionImage[2]{nullptr};
        RenderImage2* normalImage[2]{nullptr};
        RenderImage2* depthImage[2]{nullptr};
        // 6
        RenderImage2* albedoImage{nullptr};
        RenderImage2* pixelCheck{nullptr};
        RenderImage2* metallicImage{nullptr};
        RenderImage2* roughnessImage{nullptr};
        RenderImage2* objectAmbientImage{nullptr};
        // 10
        BufferBase2* generalBuffer{nullptr};
        // total 12 reqs


        CreateInfo inputData{};

        void processCopy(DeferredRender&& input) noexcept{
            if(this == &input) return;
            processDelete();
            pipe = std::move(input.pipe);
            context = std::exchange(input.context , nullptr);
            manager = std::exchange(input.manager , nullptr);
            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            for(auto& i : {0 , 1}){
                positionImage[i] = std::exchange(input.positionImage[i] , nullptr);
                normalImage[i] = std::exchange(input.normalImage[i] , nullptr);
                depthImage[i] = std::exchange(input.depthImage[i] , nullptr);
            }


            albedoImage = std::exchange(input.albedoImage , nullptr);
            pixelCheck = std::exchange(input.pixelCheck , nullptr);
            metallicImage = std::exchange(input.metallicImage , nullptr);
            roughnessImage = std::exchange(input.roughnessImage , nullptr);
            objectAmbientImage = std::exchange(input.objectAmbientImage , nullptr);
            depthBuffer = std::move(input.depthBuffer);
            completedInit = input.completedInit;
            inputData = input.inputData;
        }

        void processDelete() noexcept{
            pipe = nullptr;
            context = nullptr;
            manager = nullptr;
            generalBuffer = nullptr;
            for(auto& i : {0 , 1}){
                positionImage[i] = nullptr;
                normalImage[i] = nullptr;
                depthImage[i] = nullptr;
            }
            albedoImage = nullptr;
            pixelCheck = nullptr;
            metallicImage = nullptr;
            roughnessImage = nullptr;
            objectAmbientImage = nullptr;
            depthBuffer = nullptr;
            completedInit = false;
            inputData = {};
        }

        [[nodiscard]] ErrorDataOutput<void> processInit(const CreateInfo& input) noexcept{
            try{
                pipe = std::make_shared<ShaderPipeType>(context);

                // descriptor sets first
                {   
                    // set 0 : only the buffers , the images of the g buffer are the color attachments
                    std::vector<DescriptorLayoutData> data(5);
                    data[0] = {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ,.count = 1};   // general buffer
                    data[1] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // instance layout
                    data[2] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // mesh data
                    data[3] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // instances ( the matrices )
                    data[4] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // LUT

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

                    pipe->descriptorLayouts.resize(1);
                    pipe->descriptorLayouts[0] = std::make_shared<DescriptorLayout>(info);
                }

                {
                    PipelineLayout::CreateInfo info{};
                    info.layouts = {pipe->descriptorLayouts[0].get() , manager->layout.get()};
                    info.context = input.context;
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                // the buffers do not depend on the frame , one descriptor buffer is enough
                {
                    std::vector<DescriptorData2> data(5);
                    data[0] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                    data[1] = DescriptorData2::createData(Mesh2::instanceLayoutBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0);
                    data[2] = DescriptorData2::createData(Mesh2::MeshBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 2 , 0);
                    data[3] = DescriptorData2::createData(Mesh2::instanceBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0);
                    data[4] = DescriptorData2::createData(Mesh2::LUT_buffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->buffers.resize(1);
                    pipe->buffers[0] = std::make_shared<DescriptorBuffer2>(info);
                }
                pipe->shaders.resize(2);
                {
                    // vertex
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_VERTEX_BIT;
                    info.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
                    info.filePath = input.vertex;
                    info.layouts = {pipe->descriptorLayouts[0].get() , manager->layout.get()};
                    info.pushes = {};

                    pipe->shaders[0]=  std::make_shared<ShaderObjectType>(info);
                }

                {
                    // fragment
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_FRAGMENT_BIT;
                    info.nextStage = 0;
                    info.filePath = input.fragment;
                    info.layouts = {pipe->descriptorLayouts[0].get() , manager->layout.get()};
                    info.pushes = {};

                    pipe->shaders[1]=  std::make_shared<ShaderObjectType>(info);
                }

                {
                    RenderImage2::CreateInfo info{};
                    info.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                    info.clear.float32[0] = 1.0f;
                    info.imageFormat = VK_FORMAT_D32_SFLOAT;
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.context = input.context;
                    info.layerCount = 1;
                    info.mipMapCount = 1;
                    info.extent = input.context->sc->extent;
                    info.imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                    
                    depthBuffer = std::make_shared<RenderImage2>(info);
                }   
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create required params for Deferred Render"});
            }

            completedInit = true;
            return {};
        }

        // buffers like LUT or instance will be used directly from Mesh2
    };

    using pDeferredRender = std::shared_ptr<DeferredRender>;
}