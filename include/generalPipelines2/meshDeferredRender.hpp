#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../Object2/Mesh2.hpp"

namespace mox{

    class MeshDeferredRender : public GraphPass2{
    public:

        MeshDeferredRender() = delete;
        MeshDeferredRender(MeshDeferredRender& input) = delete;
        MeshDeferredRender(const MeshDeferredRender& input) = delete;
        MeshDeferredRender& operator=(MeshDeferredRender& input) = delete;
        MeshDeferredRender& operator=(const MeshDeferredRender& input) = delete;

        MeshDeferredRender(MeshDeferredRender&& input){
            processCopy(std::move(input));
        }

        MeshDeferredRender& operator=(MeshDeferredRender&& input){
            processCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context;
            TextureManager2* manager;

            std::string task , mesh , fragment;
        };

        // one task group checks this many meshlets of ONE instance ( it has to be the same as TASK_GROUP_SIZE in taskShader.hlsl and meshShader.hlsl )
        inline static constexpr uint32_t TASK_GROUP_SIZE = 32;

        MeshDeferredRender(const CreateInfo &info){
            if(!info.manager){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Mesh Deferred Render , since the Texture manager2 is nullptr"});
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

        ~MeshDeferredRender(){
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

            const auto bound = bindAttributes(sources);
            if(!bound.has_value()) return bound;

            return processInit(inputData);
        }

        [[nodiscard]] [[gnu::hot]]  const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translateBarriers , since the pipeline wasnt created [MeshDeferredRender]"});
            const uint8_t currentImage = context->sc->frameIndex %2;

            // the g buffer is written through the color attachments of the dynamic rendering : the load op clears them ,
            // so the old contents are not needed , and the images stay in GENERAL ( the next passes read them as storage images )
            std::array<VkImageMemoryBarrier2 , 8> barriers{};
            barriers[0] = image("albedoImage")->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[1] = image("positionImage" , currentImage)->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[2] = image("normalImage" , currentImage)->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[3] = image("depthImage" , currentImage)->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[4] = image("roughnessImage")->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[5] = image("metallicImage")->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[6] = image("pixelCheck")->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            barriers[7] = image("objectAmbientImage")->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            // the general buffer is read by the task shader , the mesh shader and the fragment shader ( no vertex stage here )
            VkBufferMemoryBarrier2 generalBufferBarrier{};
            generalBufferBarrier = buffer("generalBuffer")->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 1;
            info.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            info.pMemoryBarriers = nullptr;
            info.pBufferMemoryBarriers = &generalBufferBarrier;
            info.pImageMemoryBarriers = barriers.data();
            info.memoryBarrierCount = 0;
            info.pNext = nullptr;
            info.dependencyFlags = 0;

            vkCmdPipelineBarrier2(cmd , &info);

            return {};
        }

        [[nodiscard]] [[gnu::hot]]  ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to executePass , since the pipeline wasnt created [MeshDeferredRender]"});

            // the vertex stage is bound as VK_NULL_HANDLE : a draw with a mesh shader must not have the vertex , tessellation and geometry stages ( the old pass may have left the vertex shader bound )
            VkShaderEXT shaders[4] = {VK_NULL_HANDLE , pipe->shaders[0]->shader , pipe->shaders[1]->shader , pipe->shaders[2]->shader};
            VkShaderStageFlagBits stages[4] = {VK_SHADER_STAGE_VERTEX_BIT , VK_SHADER_STAGE_TASK_BIT_EXT , VK_SHADER_STAGE_MESH_BIT_EXT , VK_SHADER_STAGE_FRAGMENT_BIT};
            const uint8_t currentImage = context->sc->frameIndex %2;

            cmdBindShadersEXT(cmd , 4 , stages , shaders);

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
            vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
            vkCmdSetDepthTestEnable(cmd, VK_TRUE);
            vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
            vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_LESS);
            vkCmdSetViewportWithCount(cmd, 1, &viewport);
            vkCmdSetScissorWithCount(cmd, 1, &scissor);

            // no vertex input , no primitive topology , no index buffer : the mesh shader reads the vertices itself and makes the triangles

            vkCmdSetRasterizerDiscardEnable(cmd , VK_FALSE);
            vkCmdSetDepthBiasEnable(cmd , VK_FALSE);
            vkCmdSetStencilTestEnable(cmd , VK_FALSE);
            cmdSetPolygonModeEXT(cmd , VK_POLYGON_MODE_FILL);
            cmdSetRasterizationSamplesEXT(cmd , VK_SAMPLE_COUNT_1_BIT);
            const VkSampleMask sampleMask = 0xFFFFFFFF;
            cmdSetSampleMaskEXT(cmd , VK_SAMPLE_COUNT_1_BIT , &sampleMask);
            cmdSetAlphaToCoverageEnableEXT(cmd , VK_FALSE);

            // the eight color attachments : no blending , everything is written
            {
                const VkBool32 blend[8] = {VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE};
                const VkColorComponentFlags all = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                const VkColorComponentFlags mask[8] = {all , all , all , all , all , all , all , all};

                cmdSetColorBlendEnableEXT(cmd , 0 , 8 , blend);
                cmdSetColorWriteMaskEXT(cmd , 0 , 8 , mask);
            }

            // the shading rate of the primitive replaces the pipeline one ( the mesh shader does not write one : 1 x 1 )
            const VkExtent2D shadingRate = {1 , 1};
            const VkFragmentShadingRateCombinerOpKHR combiners[2] = {VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR , VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR};
            cmdSetFragmentShadingRateKHR(cmd , &shadingRate , combiners);

            // descriptor buffers : set 0 has EVERY buffer of the pass ( see processInit ) , set 1 is the textures
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
                RenderImage2* targets[8] = {image("albedoImage") , image("positionImage" , currentImage) , image("normalImage" , currentImage) , image("depthImage" , currentImage) , image("roughnessImage") , image("metallicImage") , image("pixelCheck") , image("objectAmbientImage")};
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

            uint32_t maxMeshlets = 0;
            for(const auto& geometry : Mesh2::totalGeometry){
                maxMeshlets = std::max(maxMeshlets , geometry.meshletCount);
            }
            const uint32_t groupsX = (maxMeshlets + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;
            const uint32_t groupsY = static_cast<uint32_t>(Mesh2::totalInstanceLayouts.size());
            if(groupsX > 0 && groupsY > 0){
                cmdDrawMeshTasksEXT(cmd , groupsX , groupsY , 1);
            }

            vkCmdEndRendering(cmd);

            // nothing stays bound : the next pass binds its own shaders
            VkShaderEXT none[4] = {VK_NULL_HANDLE , VK_NULL_HANDLE , VK_NULL_HANDLE , VK_NULL_HANDLE};
            cmdBindShadersEXT(cmd , 4 , stages , none);

            return {};
        }

        pRenderImage2 depthBuffer{nullptr};

    private:

        pShaderPipeType pipe{nullptr};

        VulkanContext* context{nullptr};
        TextureManager2* manager{nullptr};

        // total 12 reqs

        CreateInfo inputData{};

        void processCopy(MeshDeferredRender&& input) noexcept{
            if(this == &input) return;
            processDelete();
            GraphPass2::operator=(std::move(input));
            pipe = std::move(input.pipe);
            context = std::exchange(input.context , nullptr);
            manager = std::exchange(input.manager , nullptr);

            depthBuffer = std::move(input.depthBuffer);
            completedInit = input.completedInit;
            inputData = input.inputData;
        }

        void processDelete() noexcept{
            clearAttributes();
            pipe = nullptr;
            context = nullptr;
            manager = nullptr;
            depthBuffer = nullptr;
            completedInit = false;
            inputData = {};
        }

        [[nodiscard]] ErrorDataOutput<void> processInit(const CreateInfo& input) noexcept{
            try{
                pipe = std::make_shared<ShaderPipeType>(context);

                // ONE descriptor set for the whole pass : 0 .. 4 are the same as in DeferredRender ( the fragment shader is the same one ) , 5 .. 10 are the meshlets .
                // the numbers have to be the same as [[vk::binding(N , 0)]] in taskShader.hlsl and meshShader.hlsl
                {
                    std::vector<DescriptorLayoutData> data(11);
                    data[0]  = {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ,.count = 1};   // general buffer
                    data[1]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // instance layout
                    data[2]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // mesh data
                    data[3]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // instances ( the matrices )
                    data[4]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // LUT
                    data[5]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // geometry ( the meshlets of a mesh )
                    data[6]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // meshlets
                    data[7]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // meshlet bounds ( sphere + cone )
                    data[8]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // meshlet vertices
                    data[9]  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // meshlet indices
                    data[10] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,.count = 1};   // the vertices of all the meshes

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;

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

                // the buffers do not depend on the frame , one descriptor buffer is enough ( Mesh2::End() has to be done : the buffers must exist )
                {
                    std::vector<DescriptorData2> data(11);
                    data[0]  = DescriptorData2::createData(buffer("generalBuffer") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                    data[1]  = DescriptorData2::createData(Mesh2::instanceLayoutBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0);
                    data[2]  = DescriptorData2::createData(Mesh2::MeshBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 2 , 0);
                    data[3]  = DescriptorData2::createData(Mesh2::instanceBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0);
                    data[4]  = DescriptorData2::createData(Mesh2::LUT_buffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0);
                    data[5]  = DescriptorData2::createData(Mesh2::geometryBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 5 , 0);
                    data[6]  = DescriptorData2::createData(Mesh2::meshletBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 6 , 0);
                    data[7]  = DescriptorData2::createData(Mesh2::meshletBoundBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 7 , 0);
                    data[8]  = DescriptorData2::createData(Mesh2::meshletVertexBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 8 , 0);
                    data[9]  = DescriptorData2::createData(Mesh2::meshletIndexBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 9 , 0);
                    data[10] = DescriptorData2::createData(Mesh2::vertexBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 10 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->buffers.resize(1);
                    pipe->buffers[0] = std::make_shared<DescriptorBuffer2>(info);
                }
                pipe->shaders.resize(3);
                {
                    // task
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_TASK_BIT_EXT;
                    info.nextStage = VK_SHADER_STAGE_MESH_BIT_EXT;
                    info.filePath = input.task;
                    info.layouts = {pipe->descriptorLayouts[0].get() , manager->layout.get()};
                    info.pushes = {};

                    pipe->shaders[0]=  std::make_shared<ShaderObjectType>(info);
                }

                {
                    // mesh
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_MESH_BIT_EXT;
                    info.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
                    info.filePath = input.mesh;
                    info.layouts = {pipe->descriptorLayouts[0].get() , manager->layout.get()};
                    info.pushes = {};

                    pipe->shaders[1]=  std::make_shared<ShaderObjectType>(info);
                }

                {
                    // fragment : the same shader as in DeferredRender
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_FRAGMENT_BIT;
                    info.nextStage = 0;
                    info.filePath = input.fragment;
                    info.layouts = {pipe->descriptorLayouts[0].get() , manager->layout.get()};
                    info.pushes = {};

                    pipe->shaders[2]=  std::make_shared<ShaderObjectType>(info);
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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create required params for Mesh Deferred Render"});
            }

            completedInit = true;
            return {};
        }

        // buffers like LUT or instance will be used directly from Mesh2
    };

    using pMeshDeferredRender = std::shared_ptr<MeshDeferredRender>;
}
