#pragma once

#include "../ShaderPipe/ShaderPipe.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../../imgui/imgui.h"
#include "../../imgui/imconfig.h"
#include "../../imgui/imgui_internal.h"
#include "../../imgui/backends/imgui_impl_glfw.h"
#include "../../imgui/backends/imgui_impl_vulkan.h"

namespace mox{

    // ============================== INFORMATION ==============================
    //
    // now the imgui draw will be inside the core , the purpose of this is just to draw
    //
    // what the class does :
    //      the constructor  - the ImGui context and the glfw backend ( the callbacks of the Player are chained )
    //      the first frame  - the vulkan backend : it needs the format of finalImage , which is known only when the graph gives the image
    //                         ( postProcessGraphRenderLocations )
    //      executePass      - draws the draw data of the last ImGui::Render() over finalImage ( dynamic rendering , loadOp LOAD )
    //      the destructor   - waits for the GPU and shuts everything down
    //
    // what the CORE does every frame BEFORE the graph is executed ( the class does not touch it ) :
    //      ImGui_ImplVulkan_NewFrame();
    //      ImGui_ImplGlfw_NewFrame();
    //      ImGui::NewFrame();
    //      ... the windows of the UI ...
    //      ImGui::Render();
    //
    // ImGui has one global context : there can be only one Imgui2 ( and no ImGuiPipe ) at a time .
    // ===========================================================================
    class Imgui2 : public GraphPass2{
    public:

        SetTypicalOperators(Imgui2)
        SetTypicalTransferOperators(Imgui2)

        explicit Imgui2(VulkanContext* context){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager | VkReqTypeInstance , context);
                if(!result.has_value()){
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            if(!context->window || !context->window->window || !context->surface || !context->graphicsQueue || !context->sc){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Imgui2 , since the context has no window , surface , graphics queue or swapchain"});
                THROW_MESSAGE;
            }

            if(alive){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Imgui2 , since another ImGui already exists ( ImGui has a single global context )"});
                THROW_MESSAGE;
            }

            this->context = context;

            const auto result = createContext();
            if(!result.has_value()){
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        ~Imgui2(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(1);

            reqs[0].type = moxGraphTypeImage;
            reqs[0].name = "finalImage";
            reqs[0].imageSpecification.type = ImageType::RenderImage2;

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass , since the vulkan backend of ImGui wasnt created yet [Imgui2]"});

            // the core has made the draw data with ImGui::Render() : here it is only put into the command buffer
            ImDrawData* drawData = ImGui::GetDrawData();
            if(!drawData || drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f) return {};

            VkRenderingAttachmentInfo color{};
            color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color.pNext = nullptr;
            color.imageView = finalImage->view;
            color.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            color.resolveMode = VK_RESOLVE_MODE_NONE;
            // the UI goes over what is already in the image
            color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.pNext = nullptr;
            rendering.flags = 0;
            rendering.renderArea.offset = {0 , 0};
            rendering.renderArea.extent = finalImage->extent;
            rendering.layerCount = 1;
            rendering.viewMask = 0;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color;
            rendering.pDepthAttachment = nullptr;
            rendering.pStencilAttachment = nullptr;

            vkCmdBeginRendering(cmd , &rendering);
            ImGui_ImplVulkan_RenderDrawData(drawData , cmd);
            vkCmdEndRendering(cmd);

            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            return true;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindSources(sources , GraphSlot<RenderImage2>{"finalImage" , &finalImage});
            if(!bound.has_value()) return bound;

            // the pipeline of the UI is made for the format of this image
            const auto created = createVulkanBackend();
            if(!created.has_value()) return created;

            completedInit = true;
            return {};
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the vulkan backend of ImGui wasnt created yet [Imgui2]"});

            // the UI is drawn as the color attachment of the final image
            auto barrier = finalImage->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

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

            return {};
        }


    private:

        // only one ImGui context can live in a process
        inline static bool alive = false;

        VulkanContext* context{nullptr};
        RenderImage2* finalImage{nullptr};

        VkDevice device{nullptr};

        // the format has to live as long as the pipeline of the backend is created ( the backend reads the pointer in Init )
        VkFormat colorFormat{VK_FORMAT_UNDEFINED};

        bool contextCreated{false};
        bool glfwBackend{false};
        bool vulkanBackend{false};

        // the ImGui context and the glfw backend : they need nothing from the graph
        [[nodiscard]] ErrorDataOutput<void> createContext() noexcept{
            try{
                device = context->logicalDevice->device;

                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                contextCreated = true;
                alive = true;

                ImGuiIO& io = ImGui::GetIO();
                io.IniFilename = nullptr;
                io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

                // true = the backend installs its callbacks and CHAINS the ones that are already set ( the key and the mouse of the Player )
                if(!ImGui_ImplGlfw_InitForVulkan(context->window->window , true)){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to init the ImGui glfw backend"});
                }
                glfwBackend = true;
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the ImGui context"});
            }
            return {};
        }

        // the vulkan backend with the dynamic rendering : no render pass , the backend takes the format of the attachment
        [[nodiscard]] ErrorDataOutput<void> createVulkanBackend() noexcept{
            try{
                colorFormat = finalImage->format;

                DetailsCreateInfo details{};
                details.device = context->physicalDevice->device;
                details.surface = context->surface->surface;
                const auto family = queueFamilyDetails::queryQueueFamilyDetails(details);

                ImGui_ImplVulkan_InitInfo init{};
                init.ApiVersion = VK_API_VERSION_1_4;
                init.Instance = context->instance->instance;
                init.PhysicalDevice = context->physicalDevice->device;
                init.Device = device;
                init.QueueFamily = family.graphics.value();
                init.Queue = context->graphicsQueue->queue;
                // the backend makes its own pool of descriptors ( the font and the images that are added with ImGui_ImplVulkan_AddTexture )
                init.DescriptorPoolSize = 32;
                init.MinImageCount = std::max(2u , context->sc->imageCount);
                init.ImageCount = init.MinImageCount;

                init.UseDynamicRendering = true;
                init.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;

                if(!ImGui_ImplVulkan_Init(&init)){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to init the ImGui vulkan backend"});
                }
                vulkanBackend = true;
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the ImGui vulkan backend"});
            }
            return {};
        }

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(Imgui2&& input) noexcept{
            if(this == &input) return;
            processDelete();

            context = std::exchange(input.context , nullptr);
            finalImage = std::exchange(input.finalImage , nullptr);
            device = std::exchange(input.device , nullptr);
            colorFormat = input.colorFormat;

            // the ImGui context is global : it moves with the flags , and the moved from object must not shut it down
            contextCreated = std::exchange(input.contextCreated , false);
            glfwBackend = std::exchange(input.glfwBackend , false);
            vulkanBackend = std::exchange(input.vulkanBackend , false);
            completedInit = std::exchange(input.completedInit , false);
        }

        void processDelete() noexcept{
            // the backend frees buffers and textures that the GPU may still read
            if(device && vulkanBackend) vkDeviceWaitIdle(device);

            if(vulkanBackend) ImGui_ImplVulkan_Shutdown();
            if(glfwBackend) ImGui_ImplGlfw_Shutdown();
            if(contextCreated){
                ImGui::DestroyContext();
                alive = false;
            }

            // the image belongs to the graph , it is only forgotten
            finalImage = nullptr;
            context = nullptr;
            device = nullptr;
            vulkanBackend = false;
            glfwBackend = false;
            contextCreated = false;
            completedInit = false;
        }
    };

    using pImgui2 = std::shared_ptr<Imgui2>;
}
