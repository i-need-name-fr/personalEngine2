#pragma once

#include "include/Buffer2/GeneralBuffer.hpp"
#include "include/Image2/ViewPerLayer.hpp"
#include "include/Image2/ViewPerMip.hpp"
#include "include/Image2/FrameTargets.hpp"

#include "include/AS2/TLAS2.hpp"
#include "include/Object2/cube.hpp"
#include "include/Image2/MultiViewImage2.hpp"
#include "include/generalPipelines2/AEHDR.hpp"
#include "include/generalPipelines2/deferredRender.hpp"
#include "include/generalPipelines2/HDR.hpp"

#include "include/GraphRender2/GraphRender2.hpp"

namespace mox{

    class TempCore{
    public:

        TempCore(const TempCore& input) = delete;
        TempCore(TempCore&& input) = delete;
        TempCore& operator=(const TempCore& input) = delete;
        TempCore& operator=(TempCore&& input) = delete;

        TempCore(){
            // made once : the context takes it by reference every frame , nothing is created in the loop
            recordFunction = [this](VkCommandBuffer& cmd , const uint32_t index) -> ErrorDataOutput<void>{
                return recordCmd(cmd , index);
            };

            const auto result = init();
            if(!result.has_value()){
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        ~TempCore() noexcept {
            processDelete();
        }

        // the main loop : maxFrames = 0 means that it runs until the window is closed
        [[nodiscard]] ErrorDataOutput<void> run(const uint32_t maxFrames = 0) noexcept{
            GLFWwindow* window = context->window->window;

            double lastTime = glfwGetTime();
            uint32_t frames = 0;

            while(!glfwWindowShouldClose(window)){
                glfwPollEvents();

                const double now = glfwGetTime();
                const float dt = static_cast<float>(now - lastTime);
                lastTime = now;

                // while the menu is opened the cursor is free , and the camera must stay where it is
                if(!context->user->openedMenu) context->user->assetinput(window , dt);

                // acquire , record ( recordCmd ) , submit and present
                const auto result = context->processQueueLoop(recordFunction);
                if(!result.has_value()){
                    vkDeviceWaitIdle(context->logicalDevice->device);
                    return result;
                }

                frames++;
                if(maxFrames != 0 && frames >= maxFrames) break;
            }

            vkDeviceWaitIdle(context->logicalDevice->device);
            return {};
        }

    private:


        [[nodiscard]] ErrorDataOutput<void> init() noexcept{
            try{

                {
                    VulkanContext::CreateInfo info{};
                    info.asyncMultiThreadQueueCount = 4;
                    info.multithreadQueueCount = 1;
                    context = std::make_shared<VulkanContext>(info);
                }

                {
                    uint32_t count;
                    vkGetSwapchainImagesKHR(context->logicalDevice->device , context->sc->sc , &count , nullptr);

                    images.resize(count);
                    views.resize(count);
                    vkGetSwapchainImagesKHR(context->logicalDevice->device , context->sc->sc , &count , images.data());
                    uint32_t i = 0;
                    for(auto& image : images){
                        VkImageViewCreateInfo info{};
                        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                        info.flags = 0;
                        info.image = image;
                        info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                        info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                        info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                        info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
                        info.format = context->sc->format.format;
                        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        info.subresourceRange.baseArrayLayer = 0;
                        info.subresourceRange.baseMipLevel = 0;
                        info.subresourceRange.layerCount = 1;
                        info.subresourceRange.levelCount = 1;

                        const auto result = vkCreateImageView(context->logicalDevice->device , &info ,nullptr , &views[i]);
                        if(result != VK_SUCCESS) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create image view for swap chain , error ID -> P{}" , static_cast<int32_t>(result)) });
                        i++;
                    }
                }

                {
                    source = std::make_shared<FrameTargets>(context.get());
                }

                {
                    TextureManager2::CreateInfo info{context.get()};
                    manager = std::make_shared<TextureManager2>(info);
                }


                {
                    {
                        const auto result = Mesh2::Begin(context.get() , manager.get());
                        if(!result.has_value()) return result;
                    }

                    Mesh2::CreateInfo info{};
                    info.color = glm::vec3(0.5f);
                    info.diffusePath = "../textures/grassDiffuse.png";
                    info.depthPath = "../textures/grassDepth.png";
                    info.roughnessPath = "../textures/grassRoughness.png";
                    info.normalPath = "../textures/grassNormal.png";
                    info.needsCalcTBN = true;
                    info.inputVertices = cube::vertices;
                    info.inputIndices = cube::indices;

                    info.roughness = 0.5f;
                    info.metallic = 0.7f;
                    info.isALightObject = false;

                    mesh = std::make_shared<Mesh2>(info);

                    {
                        glm::mat4 model = glm::mat4(1.0f);
                        model = glm::translate(model , glm::vec3(0.0f , 5.0f , 0.05f));
                        mesh->addMatrix(model);
                    }

                    {
                        glm::mat4 model = glm::mat4(1.0f);
                        model = glm::translate(model , glm::vec3(5.0f , 0.0f , 0.05f));
                        model = glm::rotate(model , glm::radians(70.0f ) , glm::vec3(0.3f , 1.0f , 0.5f ));
                        mesh->addMatrix(model);
                    }

                    {
                        glm::mat4 model = glm::mat4(1.0f);
                        model = glm::translate(model , glm::vec3(0.0f , -1.0f , 5.05f));
                        model = glm::rotate(model , glm::radians(40.0f ) , glm::vec3(0.3f , 1.0f , 0.5f ));
                        mesh->addMatrix(model);
                    }

                    {
                        const auto result = Mesh2::End();
                        if(!result.has_value()) return result;
                    }
                }

                {
                    GeneralBuffer2<BufferType::Device , true>::CreateInfo info{};
                    info.context = context.get();
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                    info.size = sizeof(GeneralInfo);

                    generalBuffer = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                    info.size = sizeof(uint32_t)  * 257;

                    history = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);

                    // the size of the structure , not the size of the pointer that holds the buffer
                    info.size = sizeof(GeneralFeatures);
                    generalFeature = std::make_shared<GeneralBuffer2<BufferType::Device , true>>(info);
                }

                graph = std::make_shared<GraphRender2>();

                {
                    DeferredRender::CreateInfo info{};
                    info.context = context.get();
                    info.fragment = "../Shaders/DeferredRender/fragmentShader.spv";
                    info.vertex = "../Shaders/DeferredRender/vertexShader.spv";
                    info.manager = manager.get();

                    render = std::make_shared<DeferredRender>(info);
                }

                {
                    AEHDR::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/HDR/autoExposure.spv";

                    aehdr = std::make_shared<AEHDR>(info);
                }

                {
                    HDR::CreateInfo info{};
                    info.context = context.get();
                    info.path =  "../Shaders/HDR/HDR_mapped.spv";
                    hdr = std::make_shared<HDR>(info);
                }

            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create all the resources"});
            }

            {
                const auto result = graph->processInputParameter(source.get());
                if(!result.has_value()) return result;
            }

            {
                const auto result = graph->processInputParameter(generalBuffer.get() , "generalBuffer");
                if(!result.has_value()) return result;
            }

            {
                const auto result = graph->processInputParameter(generalFeature.get() , "generalFeatures");
                if(!result.has_value()) return result;
            }

            {
                const auto result = graph->processInputParameter(history.get() , "luminanceHistory");
                if(!result.has_value()) return result;
            }

            {
                const auto result = graph->addGraphPass2(render.get() , "main");
                if(!result.has_value()) return result;
            }

            {
                const auto result = graph->addGraphPass2(aehdr.get() , "main");
                if(!result.has_value()) return result;
            }

            {
                const auto result = graph->addGraphPass2(hdr.get() , "main");
                if(!result.has_value()) return result;
            }

            return {};
        }

        void processDelete() noexcept {
            if(context && context->logicalDevice) vkDeviceWaitIdle(context->logicalDevice->device);

            // the graph keeps raw pointers to the passes , and the passes to the resources , so they go first
            graph = nullptr;
            hdr = nullptr;
            aehdr = nullptr;
            render=  nullptr;

            generalBuffer = nullptr;
            generalFeature = nullptr;
            history = nullptr;
            mesh = nullptr;
            Mesh2::performDelete();
            manager = nullptr;
            source = nullptr;

            // the images belong to the swapchain and are destroyed with it , only the views are ours
            if(context && context->logicalDevice){
                for(auto& view : views){
                    if(view) vkDestroyImageView(context->logicalDevice->device , view , nullptr);
                }
            }

            images.clear();
            views.clear();
            context = nullptr;
        }

        // update data
        void updateData(VkCommandBuffer &cmd) noexcept{

            const float aspect = static_cast<float>(context->sc->extent.width) / static_cast<float>(context->sc->extent.height);
            glm::mat4 proj = glm::perspective(glm::radians(70.0f) , aspect , 0.001f , 200.0f);
            glm::mat4 view = glm::lookAt(context->user->position , context->user->position + context->user->direction , glm::vec3(0.0f , 1.0f , 0.0f));

            GeneralInfo info{};
            // the frame counter is increased by the context at the end of the frame , the passes ping-pong by it
            info.frameIndex = context->sc->frameIndex;
            info.viewOnly = view;
            info.projOnly = proj;
            info.viewMode = context->user->currentMode;
            info.exposure = 1.6f;
            info.inverseView = glm::inverse(view);
            info.inverseProj = glm::inverse(proj);
            info.invViewProj = glm::inverse(proj * view);
            info.viewProj = proj * view;
            if(firstIteration){
                firstIteration = false;
                info.prevViewProj = info.viewProj;
            }else{
                info.prevViewProj = prevVP;
            }

            info.lightCount = 0;
            info.playerDirection = context->user->direction;
            info.playerFar = 200.0f;
            info.playerNear = 0.001f;
            info.screenHeight = context->sc->extent.height;
            info.screenWidth = context->sc->extent.width;
            info.playerPosition = context->user->position;

            prevVP = proj * view;

            generalBuffer->updateBufferCmd(cmd , info , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

            GeneralFeatures features{};
            features.maxDepth = 50.0f;
            features.currentMode = info.viewMode;

            generalFeature->updateBufferCmd(cmd , features , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        }

        // the body of a frame : the context calls it between the acquire of the image and the submit
        [[nodiscard]] ErrorDataOutput<void> recordCmd(VkCommandBuffer &cmd , uint32_t index) noexcept{
            {
                VkCommandBufferBeginInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                info.pInheritanceInfo = nullptr;
                info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                info.pNext = nullptr;

                const auto result = vkBeginCommandBuffer(cmd , &info);
                if(result != VK_SUCCESS) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to begin the command buffer , error ID -> {}" , static_cast<int32_t>(result))});
            }

            updateData(cmd);
            source->cleanImages(cmd);
            {
                const auto result = graph->execute(cmd , "main");
                if(!result.has_value()){
                    vkEndCommandBuffer(cmd);
                    return result;
                }
            }

            VkImageMemoryBarrier2 swapchainBarrier{};
            swapchainBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            swapchainBarrier.pNext = nullptr;
            swapchainBarrier.image = images[index];
            swapchainBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT , 0 , 1 , 0 , 1};

            // the final image is read by the copy , the swapchain image is written by it
            {
                std::array<VkImageMemoryBarrier2, 2> barriers;
                barriers[0] = source->finalImage->translateImage(VK_ACCESS_2_TRANSFER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

                barriers[1] = swapchainBarrier;
                barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barriers[1].srcAccessMask = 0;
                barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                barriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 2;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = barriers.data();
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);
            }

            {
                VkImageCopy region{};
                region.extent = {context->sc->extent.width , context->sc->extent.height , 1};
                region.dstOffset = {0 , 0};
                region.srcOffset = {0 , 0};
                region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.srcSubresource.baseArrayLayer = 0;
                region.srcSubresource.layerCount = 1;
                region.srcSubresource.mipLevel = 0;
                region.dstSubresource = region.srcSubresource;

                vkCmdCopyImage(cmd , source->finalImage->image , VK_IMAGE_LAYOUT_GENERAL, images[index] , VK_IMAGE_LAYOUT_GENERAL , 1 , &region);
            }

            // the swapchain image is presented , and only PRESENT_SRC is a layout that can be presented
            {
                VkImageMemoryBarrier2 barrier = swapchainBarrier;
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                barrier.dstAccessMask = 0;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

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

            const auto result = vkEndCommandBuffer(cmd);
            if(result != VK_SUCCESS) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to end the command buffer , error ID -> {}" , static_cast<int32_t>(result))});

            return {};
        }

        pFrameTargets source{}; // created
        pVulkanContext context{}; // created
        pGeneralBuffer2<BufferType::Device , true> generalBuffer; // created
        pGeneralBuffer2<BufferType::Device , true> generalFeature; // created
        pGeneralBuffer2<BufferType::Device , true> history; // created
        pTextureManager2 manager; // created

        pMesh2 mesh; // created
        std::vector<VkImage> images{}; // created
        std::vector<VkImageView> views{}; // created

        pHDR  hdr{}; // created
        pAEHDR aehdr{}; // created
        pDeferredRender render{}; // created

        glm::mat4 prevVP{1.0f};
        bool firstIteration{true};

        QueueRecordTypeFunction recordFunction{};

        pGraphRender2 graph{};
    };

}
