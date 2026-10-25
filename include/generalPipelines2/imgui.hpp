#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../../imgui/imgui.h"
#include "../../imgui/imconfig.h"
#include "../../imgui/imgui_internal.h"
#include "../../imgui/backends/imgui_impl_glfw.h"
#include "../../imgui/backends/imgui_impl_vulkan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <string>

namespace mox{

    // ============================== INFORMATION ==============================
    //
    // the ImGui overlay as the LAST pass of the graph : it takes the finalImage ( and only it ) and draws the UI over it with the dynamic rendering .
    //
    // the data of the shaders : the five void* of the CreateInfo are the DeviceReadableBuffer* ( GeneralBuffer2<Device , true> ) of
    //      generalData  - GeneralInfo
    //      features     - GeneralFeatures
    //      fogData      - FogVolume
    //      sunData      - SunSource
    //      skyViewData  - SkyViewData
    // the UI reads the CPU copy of the buffer ( getBuffetDataSingular ) every frame , and when a value really changed it writes it back : to the CPU copy
    // ( recordData ) and to the GPU ( updateBufferCmd ) . WHO OWNS A BUFFER MUST NOT REWRITE THE FIELDS OF THE UI FROM SCRATCH EVERY FRAME ( tempCore builds a new
    // GeneralInfo every frame : it has to take the CPU copy of the buffer and change only its own fields , the matrices , the frame index ... ) .
    //
    // the derived values : when the sun moves the UI recomputes the direction of the light , cos of the zenith , cos and sin of the azimuth of the sky view data
    // ( the sky view LUT always has the sun at the azimuth 0 , the cube turns it ) . the height of the camera is not a thing of the UI .
    //
    // the rules of the features ( BeginDisabled ) :
    //      path tracing ( PT )  : no AO , no VSM , no RayQuery shadows , no SSR / SSGI , and no RT ( PT replaces it )
    //      ray tracing ( RT )   : no VSM , no RayQuery shadows , no SSR / SSGI
    //      what was chosen in the raster mode is remembered and comes back when the raster mode is selected again .
    //
    // the menu is opened with F ( Player::keyCallBack ) , the small HUD stays on the screen .
    // ImGui has one global context : there can be only one ImGuiPipe at a time .
    // ===========================================================================
    class ImGuiPipe : public GraphPass2{
    public:

        SetTypicalOperators(ImGuiPipe)
        SetTypicalTransferOperators(ImGuiPipe)

        struct CreateInfo{
            VulkanContext* context;
            // all of these are DeviceReadableBuffer* : the UI reads them and writes the changes back
            void* generalData;
            void* features;
            void* fogData;
            void* sunData;
            void* skyViewData;
        };

        explicit ImGuiPipe(const CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager | VkReqTypeInstance , info.context);
                if(!result.has_value()){
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            if(!info.context->window || !info.context->window->window || !info.context->surface || !info.context->graphicsQueue || !info.context->sc){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create ImGuiPipe , since the context has no window , surface , graphics queue or swapchain"});
                THROW_MESSAGE;
            }

            if(!info.generalData || !info.features || !info.fogData || !info.sunData || !info.skyViewData){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create ImGuiPipe , since one of the data pointers ( generalData , features , fogData , sunData , skyViewData ) is nullptr"});
                THROW_MESSAGE;
            }

            if(alive){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create ImGuiPipe , since another ImGuiPipe already exists ( ImGui has a single global context )"});
                THROW_MESSAGE;
            }

            // void* has no type to check : the owner of the pointers promises that these are DeviceReadableBuffer ( the CPU has to read them )
            generalBuffer = std::bit_cast<DeviceReadableBuffer*>(info.generalData);
            featuresBuffer = std::bit_cast<DeviceReadableBuffer*>(info.features);
            fogBuffer = std::bit_cast<DeviceReadableBuffer*>(info.fogData);
            sunBuffer = std::bit_cast<DeviceReadableBuffer*>(info.sunData);
            skyBuffer = std::bit_cast<DeviceReadableBuffer*>(info.skyViewData);

            inputData = info;
        }

        ~ImGuiPipe(){
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
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since ImGui wasnt created yet [ImGuiPipe]"});

            // the CPU copies of this frame
            Data data = readData();
            const Data before = data;

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            drawUI(data);

            ImGui::Render();

            // the values that follow from the values of the UI ( the azimuth of the sun ... )
            deriveValues(data);

            ImDrawData* drawData = ImGui::GetDrawData();
            if(drawData && drawData->DisplaySize.x > 0.0f && drawData->DisplaySize.y > 0.0f){

                VkRenderingAttachmentInfo color{};
                color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                color.pNext = nullptr;
                color.imageView = image("finalImage")->view;
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
                rendering.renderArea.extent = image("finalImage")->extent;
                rendering.layerCount = 1;
                rendering.viewMask = 0;
                rendering.colorAttachmentCount = 1;
                rendering.pColorAttachments = &color;
                rendering.pDepthAttachment = nullptr;
                rendering.pStencilAttachment = nullptr;

                vkCmdBeginRendering(cmd , &rendering);
                ImGui_ImplVulkan_RenderDrawData(drawData , cmd);
                vkCmdEndRendering(cmd);
            }

            // the changes go to the CPU copy and to the GPU ( outside of the rendering )
            commitData(cmd , before , data);

            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            return true;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindAttributes(sources);
            if(!bound.has_value()) return bound;

            const auto created = createImGui();
            if(!created.has_value()) return created;

            completedInit = true;
            return {};
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since ImGui wasnt created yet [ImGuiPipe]"});

            // the UI is drawn as the color attachment of the final image
            auto barrier = image("finalImage")->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

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

        // ------------------------------------------------------------------ state

        inline static bool alive = false;

        CreateInfo inputData{};

        // the buffers of the shaders : the UI reads their CPU copy and writes the changes back
        DeviceReadableBuffer* generalBuffer{nullptr};
        DeviceReadableBuffer* featuresBuffer{nullptr};
        DeviceReadableBuffer* fogBuffer{nullptr};
        DeviceReadableBuffer* sunBuffer{nullptr};
        DeviceReadableBuffer* skyBuffer{nullptr};

        // the format has to live as long as the pipeline of the backend is created ( it is read by the backend in Init )
        VkFormat colorFormat{VK_FORMAT_UNDEFINED};
        VkDevice device{nullptr};

        bool contextCreated{false};
        bool glfwBackend{false};
        bool vulkanBackend{false};

        // what the buffers held when the UI started : the reset buttons go back to it
        GeneralFeatures defaultFeatures{};
        FogVolume defaultFog{};
        SunSource defaultSun{};
        float defaultIndirectWeight{0.5f};

        // the choice of the raster mode : it is remembered while a mode that forbids it is selected
        struct RasterChoice{
            uint32_t AO{0};
            uint32_t indirectMode{0};
            uint32_t VSM{0};
            uint32_t RayQuery{0};
        } rasterChoice{};

        // the sun as the UI sees it : the angles and the color ( the buffer holds the direction of the light and the diffuse color )
        float sunElevation{1.0f};
        float sunAzimuth{0.0f};
        float sunIntensity{1.0f};
        glm::vec3 sunColor{1.0f};
        bool sunMoved{false};

        // the fog as the UI sees it : the color of the glow and its intensity ( the buffer holds the product , the intensity can go to 0 without losing the color )
        glm::vec3 fogColor{0.55f , 0.65f , 0.85f};
        float fogIntensity{0.0f};

        // frame time history for the graph ( ms )
        static constexpr int HISTORY = 160;
        std::array<float , HISTORY> frameTimes{};
        int frameOffset{0};
        int frameFilled{0};
        float graphMax{16.6f};

        // 0 = hidden , 1 = fully open , eased every frame
        float menuAnim{0.0f};

        // all the data that the UI edits in one frame
        struct Data{
            GeneralInfo general;
            GeneralFeatures features;
            FogVolume fog;
            SunSource sun;
            SkyViewData sky;
        };

        [[nodiscard]] Data readData() const noexcept{
            Data data{};
            data.general = generalBuffer->getBuffetDataSingular<GeneralInfo>();
            data.features = featuresBuffer->getBuffetDataSingular<GeneralFeatures>();
            data.fog = fogBuffer->getBuffetDataSingular<FogVolume>();
            data.sun = sunBuffer->getBuffetDataSingular<SunSource>();
            data.sky = skyBuffer->getBuffetDataSingular<SkyViewData>();
            return data;
        }

        template<typename T>
        static void commitOne(VkCommandBuffer& cmd , DeviceReadableBuffer* buffer , const T& before , const T& after) noexcept{
            if(std::memcmp(&before , &after , sizeof(T)) == 0) return;

            T value = after;
            // the buffer is HOST_VISIBLE | HOST_COHERENT : the CPU copy IS the memory that the shaders read , so one write is enough . a vkCmdUpdateBuffer on top of it
            // is executed later , when the GPU gets to this frame , and it would put the old value back over what the CPU wrote in a newer frame
            // ( the checkbox of the fog jumped back , the mode of the view came late ) . the write is visible to the next submit
            buffer->recordData(&value , sizeof(T));
            (void)cmd;
        }

        void commitData(VkCommandBuffer& cmd , const Data& before , const Data& after) noexcept{
            commitOne(cmd , generalBuffer , before.general , after.general);
            commitOne(cmd , featuresBuffer , before.features , after.features);
            commitOne(cmd , fogBuffer , before.fog , after.fog);
            commitOne(cmd , sunBuffer , before.sun , after.sun);
            commitOne(cmd , skyBuffer , before.sky , after.sky);
        }

        // ------------------------------------------------------------------ the derived values

        // the direction to the sun from the angles ( Y is up ) : the elevation above the horizon and the azimuth around Y ( 0 is +X )
        static glm::vec3 directionToSun(const float elevation , const float azimuth) noexcept{
            return glm::vec3(std::cos(elevation) * std::cos(azimuth) , std::sin(elevation) , std::cos(elevation) * std::sin(azimuth));
        }

        // everything that follows from the sun : the light of the sun ( travels FROM the sun : minus the direction to it ) and the angles of the sky view data
        void deriveValues(Data& data) noexcept{
            if(sunMoved){
                const glm::vec3 toSun = directionToSun(sunElevation , sunAzimuth);
                data.sun.direction = -toSun;
                data.sun.intensity = sunIntensity;
                data.sun.diffuse = sunColor * sunIntensity;
                sunMoved = false;
            }

            // the sky view data from the light that is in the buffer now ( it is right when the sun was moved by something else too )
            const glm::vec3 toSun = -glm::normalize(glm::vec3(data.sun.direction));
            // cos of the angle between the sun and the zenith ( up is Y )
            data.sky.sunCosZenith = std::clamp(toSun.y , -1.0f , 1.0f);

            // the horizontal direction of the sun : the sky view LUT has the sun at the azimuth 0 , the cube is turned by atan2( sin , cos ) of this
            const float horizontal = std::sqrt(toSun.x * toSun.x + toSun.z * toSun.z);
            if(horizontal > 1e-5f){
                data.sky.sunCosAzimuth = toSun.x / horizontal;
                data.sky.sunSinAzimuth = toSun.z / horizontal;
            }else{
                // the sun is straight above : any azimuth is right , the old one is kept so the sky does not turn
                const float oldLength = std::sqrt(data.sky.sunCosAzimuth * data.sky.sunCosAzimuth + data.sky.sunSinAzimuth * data.sky.sunSinAzimuth);
                if(oldLength < 1e-5f){
                    data.sky.sunCosAzimuth = 1.0f;
                    data.sky.sunSinAzimuth = 0.0f;
                }
            }
        }

        // ------------------------------------------------------------------ the rules of the features

        enum class RenderMode : int{
            Raster = 0,
            RayTracing = 1,
            PathTracing = 2
        };

        // what is locked in the mode : BeginDisabled of the UI and the forced zeros of the data
        struct Locks{
            bool ao;
            bool shadows;
            bool indirect;
            bool rayTracing;
        };

        static RenderMode modeOf(const GeneralFeatures& f) noexcept{
            if(f.PT) return RenderMode::PathTracing;
            if(f.RT) return RenderMode::RayTracing;
            return RenderMode::Raster;
        }

        static Locks locksOf(const RenderMode mode) noexcept{
            Locks locks{};
            // the path tracer computes the occlusion itself
            locks.ao = (mode == RenderMode::PathTracing);
            // VSM and RayQuery shadows : the traced light has its own shadows
            locks.shadows = (mode != RenderMode::Raster);
            // SSR / SSGI : the traced light has its own reflections and bounces
            locks.indirect = (mode != RenderMode::Raster);
            // PT replaces RT
            locks.rayTracing = (mode == RenderMode::PathTracing);
            return locks;
        }

        // the mode is changed : the choice of the raster mode is restored , and what the new mode locks is turned off
        void applyMode(GeneralFeatures& f , const RenderMode mode) noexcept{
            f.AO = rasterChoice.AO;
            f.indirectMode = rasterChoice.indirectMode;
            f.VSM = rasterChoice.VSM;
            f.RayQuery = rasterChoice.RayQuery;

            f.RT = (mode == RenderMode::RayTracing) ? 1u : 0u;
            f.PT = (mode == RenderMode::PathTracing) ? 1u : 0u;

            enforceLocks(f);

            auto& player = inputData.context->user;
            if(player) player->PT_required = (mode == RenderMode::PathTracing);
        }

        // the locked features are off in the data : the passes and the shaders read these flags
        static void enforceLocks(GeneralFeatures& f) noexcept{
            const Locks locks = locksOf(modeOf(f));
            if(locks.ao) f.AO = AO_NOT_REQUIRED;
            if(locks.shadows){
                f.VSM = 0;
                f.RayQuery = 0;
            }
            if(locks.indirect) f.indirectMode = NON_REQUIRED;
        }

        // only the raster mode keeps the choices : in the other ones the locked features are zeros in the data
        void rememberRasterChoice(const GeneralFeatures& f) noexcept{
            if(modeOf(f) != RenderMode::Raster) return;
            rasterChoice.AO = f.AO;
            rasterChoice.indirectMode = f.indirectMode;
            rasterChoice.VSM = f.VSM;
            rasterChoice.RayQuery = f.RayQuery;
        }

        // ------------------------------------------------------------------ init

        [[nodiscard]] ErrorDataOutput<void> createImGui() noexcept{
            try{
                VulkanContext* context = inputData.context;
                device = context->logicalDevice->device;
                colorFormat = image("finalImage")->format;

                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                contextCreated = true;
                alive = true;

                ImGuiIO& io = ImGui::GetIO();
                io.IniFilename = nullptr;
                io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

                applyTheme();

                // true = the backend installs its callbacks and CHAINS the ones that are already set ( the key and the mouse of the Player )
                if(!ImGui_ImplGlfw_InitForVulkan(context->window->window , true)){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to init the ImGui glfw backend"});
                }
                glfwBackend = true;

                DetailsCreateInfo details{};
                details.device = context->physicalDevice->device;
                details.surface = context->surface->surface;
                const auto family = queueFamilyDetails::queryQueueFamilyDetails(details);

                ImGui_ImplVulkan_InitInfo init{};
                init.ApiVersion = VK_API_VERSION_1_3;
                init.Instance = context->instance->instance;
                init.PhysicalDevice = context->physicalDevice->device;
                init.Device = device;
                init.QueueFamily = family.graphics.value();
                init.Queue = context->graphicsQueue->queue;
                init.DescriptorPoolSize = 32;
                init.MinImageCount = std::max(2u , context->sc->imageCount);
                init.ImageCount = init.MinImageCount;

                // the dynamic rendering : the pipeline of the UI is made for the format of the final image
                init.UseDynamicRendering = true;
                init.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;

                if(!ImGui_ImplVulkan_Init(&init)){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to init the ImGui vulkan backend"});
                }
                vulkanBackend = true;

                // what the buffers hold now is what the reset buttons go back to , and the angles of the sun are taken from the light
                const Data data = readData();
                defaultFeatures = data.features;
                defaultFog = data.fog;
                defaultSun = data.sun;
                defaultIndirectWeight = data.general.indirectWeight;
                rememberRasterChoice(data.features);
                sunFromData(data.sun);
                fogFromData(data.fog);
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the ImGui context"});
            }
            return {};
        }

        // the color and the intensity of the glow of the fog from the buffer ( the color stays when the intensity is 0 )
        void fogFromData(const FogVolume& fog) noexcept{
            const glm::vec3 color = glm::vec3(fog.color);
            const float peak = std::max({color.x , color.y , color.z});
            if(peak > 1e-6f){
                fogIntensity = peak;
                fogColor = color / peak;
            }else{
                fogIntensity = 0.0f;
            }
        }

        // the angles , the intensity and the color of the sun from the light that is in the buffer
        void sunFromData(const SunSource& sun) noexcept{
            const glm::vec3 direction = glm::vec3(sun.direction);
            const float length = glm::length(direction);
            if(length > 1e-5f){
                const glm::vec3 toSun = -direction / length;
                sunElevation = std::asin(std::clamp(toSun.y , -1.0f , 1.0f));
                sunAzimuth = std::atan2(toSun.z , toSun.x);
            }

            const glm::vec3 diffuse = glm::vec3(sun.diffuse);
            const float peak = std::max({diffuse.x , diffuse.y , diffuse.z});
            if(peak > 1e-6f){
                sunIntensity = peak;
                sunColor = diffuse / peak;
            }else{
                sunIntensity = 0.0f;
                sunColor = glm::vec3(1.0f);
            }
        }

        // the global context belongs to whoever has these flags now
        void processCopy(ImGuiPipe&& input) noexcept{
            if(this == &input) return;
            processDelete();
            GraphPass2::operator=(std::move(input));

            inputData = input.inputData;

            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            featuresBuffer = std::exchange(input.featuresBuffer , nullptr);
            fogBuffer = std::exchange(input.fogBuffer , nullptr);
            sunBuffer = std::exchange(input.sunBuffer , nullptr);
            skyBuffer = std::exchange(input.skyBuffer , nullptr);

            colorFormat = input.colorFormat;
            device = std::exchange(input.device , nullptr);

            contextCreated = std::exchange(input.contextCreated , false);
            glfwBackend = std::exchange(input.glfwBackend , false);
            vulkanBackend = std::exchange(input.vulkanBackend , false);

            defaultFeatures = input.defaultFeatures;
            defaultFog = input.defaultFog;
            defaultSun = input.defaultSun;
            defaultIndirectWeight = input.defaultIndirectWeight;
            rasterChoice = input.rasterChoice;

            sunElevation = input.sunElevation;
            sunAzimuth = input.sunAzimuth;
            sunIntensity = input.sunIntensity;
            sunColor = input.sunColor;
            sunMoved = input.sunMoved;
            fogColor = input.fogColor;
            fogIntensity = input.fogIntensity;

            frameTimes = input.frameTimes;
            frameOffset = input.frameOffset;
            frameFilled = input.frameFilled;
            graphMax = input.graphMax;
            menuAnim = input.menuAnim;

            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            clearAttributes();
            // the backend frees buffers and textures that the GPU may still read
            if(device && vulkanBackend) vkDeviceWaitIdle(device);

            if(vulkanBackend) ImGui_ImplVulkan_Shutdown();
            if(glfwBackend) ImGui_ImplGlfw_Shutdown();
            if(contextCreated){
                ImGui::DestroyContext();
                alive = false;
            }

            // the buffers and the image belong to the owners , they are only forgotten
            generalBuffer = nullptr;
            featuresBuffer = nullptr;
            fogBuffer = nullptr;
            sunBuffer = nullptr;
            skyBuffer = nullptr;

            vulkanBackend = false;
            glfwBackend = false;
            contextCreated = false;
            device = nullptr;
            inputData = {};
            completedInit = false;
        }

        // ------------------------------------------------------------------ look

        void applyTheme() noexcept{
            ImGuiStyle& s = ImGui::GetStyle();

            s.FontSizeBase = 17.0f;

            s.WindowPadding = ImVec2(16.0f , 14.0f);
            s.FramePadding = ImVec2(10.0f , 6.0f);
            s.ItemSpacing = ImVec2(10.0f , 9.0f);
            s.ItemInnerSpacing = ImVec2(8.0f , 6.0f);
            s.ScrollbarSize = 12.0f;
            s.GrabMinSize = 14.0f;

            s.WindowRounding = 14.0f;
            s.ChildRounding = 10.0f;
            s.FrameRounding = 8.0f;
            s.PopupRounding = 10.0f;
            s.ScrollbarRounding = 10.0f;
            s.GrabRounding = 8.0f;
            s.TabRounding = 8.0f;

            s.WindowBorderSize = 1.0f;
            s.FrameBorderSize = 0.0f;
            s.PopupBorderSize = 1.0f;

            s.WindowTitleAlign = ImVec2(0.5f , 0.5f);

            ImVec4* c = s.Colors;
            const ImVec4 accent = ImVec4(0.18f , 0.80f , 0.72f , 1.00f);
            const ImVec4 accentDim = ImVec4(0.12f , 0.52f , 0.48f , 1.00f);

            c[ImGuiCol_Text] = ImVec4(0.92f , 0.94f , 0.96f , 1.00f);
            c[ImGuiCol_TextDisabled] = ImVec4(0.46f , 0.50f , 0.56f , 1.00f);
            c[ImGuiCol_WindowBg] = ImVec4(0.055f , 0.062f , 0.082f , 0.94f);
            c[ImGuiCol_ChildBg] = ImVec4(0.080f , 0.090f , 0.115f , 0.70f);
            c[ImGuiCol_PopupBg] = ImVec4(0.07f , 0.08f , 0.10f , 0.97f);
            c[ImGuiCol_Border] = ImVec4(0.20f , 0.24f , 0.30f , 0.60f);
            c[ImGuiCol_FrameBg] = ImVec4(0.11f , 0.13f , 0.17f , 1.00f);
            c[ImGuiCol_FrameBgHovered] = ImVec4(0.15f , 0.18f , 0.23f , 1.00f);
            c[ImGuiCol_FrameBgActive] = ImVec4(0.18f , 0.22f , 0.28f , 1.00f);
            c[ImGuiCol_TitleBg] = ImVec4(0.055f , 0.062f , 0.082f , 1.00f);
            c[ImGuiCol_TitleBgActive] = ImVec4(0.07f , 0.08f , 0.105f , 1.00f);
            c[ImGuiCol_CheckMark] = accent;
            c[ImGuiCol_SliderGrab] = accent;
            c[ImGuiCol_SliderGrabActive] = ImVec4(0.45f , 0.95f , 0.88f , 1.00f);
            c[ImGuiCol_Button] = ImVec4(0.13f , 0.16f , 0.21f , 1.00f);
            c[ImGuiCol_ButtonHovered] = accentDim;
            c[ImGuiCol_ButtonActive] = accent;
            c[ImGuiCol_Header] = ImVec4(0.13f , 0.16f , 0.21f , 1.00f);
            c[ImGuiCol_HeaderHovered] = ImVec4(0.16f , 0.22f , 0.28f , 1.00f);
            c[ImGuiCol_HeaderActive] = accentDim;
            c[ImGuiCol_Separator] = ImVec4(0.20f , 0.24f , 0.30f , 0.70f);
            c[ImGuiCol_Tab] = ImVec4(0.09f , 0.11f , 0.14f , 1.00f);
            c[ImGuiCol_TabHovered] = accentDim;
            c[ImGuiCol_TabSelected] = ImVec4(0.14f , 0.34f , 0.34f , 1.00f);
            c[ImGuiCol_TabSelectedOverline] = accent;
            c[ImGuiCol_TabDimmed] = ImVec4(0.08f , 0.09f , 0.12f , 1.00f);
            c[ImGuiCol_TabDimmedSelected] = ImVec4(0.11f , 0.24f , 0.25f , 1.00f);
            c[ImGuiCol_PlotLines] = accent;
            c[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f , 0.78f , 0.30f , 1.00f);
            c[ImGuiCol_PlotHistogram] = accent;
            c[ImGuiCol_ScrollbarBg] = ImVec4(0.0f , 0.0f , 0.0f , 0.0f);
            c[ImGuiCol_ScrollbarGrab] = ImVec4(0.20f , 0.24f , 0.30f , 1.00f);
        }

        static ImVec4 fpsColor(const float fps) noexcept{
            if(fps >= 100.0f) return ImVec4(0.30f , 0.90f , 0.50f , 1.0f);
            if(fps >= 60.0f) return ImVec4(0.65f , 0.90f , 0.35f , 1.0f);
            if(fps >= 30.0f) return ImVec4(1.00f , 0.70f , 0.25f , 1.0f);
            return ImVec4(1.00f , 0.35f , 0.35f , 1.0f);
        }

        // a small rounded pill , used for the "what is active right now" rows
        static void chip(const char* text , const ImVec4 color) noexcept{
            ImDrawList* list = ImGui::GetWindowDrawList();
            const ImVec2 position = ImGui::GetCursorScreenPos();
            const ImVec2 textSize = ImGui::CalcTextSize(text);
            const float padX = 9.0f;
            const float padY = 3.0f;
            const ImVec2 size(textSize.x + padX * 2.0f , textSize.y + padY * 2.0f);
            const ImVec2 end(position.x + size.x , position.y + size.y);

            list->AddRectFilled(position , end , ImGui::GetColorU32(ImVec4(color.x , color.y , color.z , 0.20f)) , size.y * 0.5f);
            list->AddRect(position , end , ImGui::GetColorU32(color) , size.y * 0.5f , 0 , 1.2f);
            list->AddText(ImVec2(position.x + padX , position.y + padY) , ImGui::GetColorU32(color) , text);
            ImGui::Dummy(size);
        }

        static void helpMarker(const char* text) noexcept{
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if(ImGui::BeginItemTooltip()){
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
                ImGui::TextUnformatted(text);
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
        }

        // a dimmed paragraph that wraps inside the window
        static void hint(const char* text) noexcept{
            ImGui::PushStyleColor(ImGuiCol_Text , ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            ImGui::TextWrapped("%s" , text);
            ImGui::PopStyleColor();
        }

        // the sky with the horizon and the sun on the arc : the elevation is the height of the sun , the azimuth is turned to the right and to the left
        static void sunPreview(const float elevation , const float azimuth , const glm::vec3 color , const float intensity) noexcept{
            const float width = ImGui::GetContentRegionAvail().x;
            const float height = 96.0f;
            const ImVec2 position = ImGui::GetCursorScreenPos();
            const ImVec2 end(position.x + width , position.y + height);
            ImDrawList* list = ImGui::GetWindowDrawList();

            const float day = std::clamp(std::sin(elevation) * 2.0f + 0.35f , 0.0f , 1.0f);
            const ImU32 top = IM_COL32(static_cast<int>(10 + 40 * day) , static_cast<int>(16 + 90 * day) , static_cast<int>(40 + 160 * day) , 255);
            const ImU32 bottom = IM_COL32(static_cast<int>(18 + 190 * (1.0f - std::abs(std::sin(elevation))) * day) , static_cast<int>(22 + 110 * day) , static_cast<int>(50 + 90 * day) , 255);
            list->AddRectFilledMultiColor(position , end , top , top , bottom , bottom);

            const float horizonY = position.y + height * 0.80f;
            list->AddRectFilled(ImVec2(position.x , horizonY) , end , IM_COL32(14 , 16 , 22 , 255));
            list->AddLine(ImVec2(position.x , horizonY) , ImVec2(end.x , horizonY) , IM_COL32(255 , 255 , 255 , 60));

            // the position on the screen : the azimuth goes along , the elevation goes up
            const float x = position.x + (std::fmod(azimuth / (2.0f * 3.14159265f) + 1.0f , 1.0f)) * width;
            const float y = horizonY - std::sin(elevation) * (horizonY - position.y - 8.0f);
            const float glow = 10.0f + std::min(intensity , 10.0f) * 1.6f;
            for(int i = 6 ; i >= 1 ; i--){
                const float t = static_cast<float>(i) / 6.0f;
                list->AddCircleFilled(ImVec2(x , y) , glow * t * 1.8f , IM_COL32(static_cast<int>(color.x * 255) , static_cast<int>(color.y * 255) , static_cast<int>(color.z * 255) , static_cast<int>((1.0f - t) * (1.0f - t) * 110)) , 28);
            }
            list->AddCircleFilled(ImVec2(x , y) , 6.0f , IM_COL32(static_cast<int>(color.x * 255) , static_cast<int>(color.y * 255) , static_cast<int>(color.z * 255) , 255) , 20);

            ImGui::Dummy(ImVec2(width , height));
        }

        // ------------------------------------------------------------------ UI

        void drawUI(Data& data) noexcept{
            ImGuiIO& io = ImGui::GetIO();

            // ---- frame time history
            const float frameMs = io.DeltaTime * 1000.0f;
            frameTimes[frameOffset] = frameMs;
            frameOffset = (frameOffset + 1) % HISTORY;
            frameFilled = std::min(frameFilled + 1 , HISTORY);

            float sum = 0.0f;
            float worst = 0.0f;
            float best = 1e9f;
            for(int i = 0 ; i < frameFilled ; i++){
                const float t = frameTimes[i];
                sum += t;
                worst = std::max(worst , t);
                if(t > 0.0f) best = std::min(best , t);
            }
            const float average = sum / static_cast<float>(std::max(frameFilled , 1));
            // eased , so the graph does not rescale every single frame
            graphMax += (std::max(worst * 1.15f , 8.0f) - graphMax) * 0.05f;

            auto& player = inputData.context->user;
            const bool menuOpen = player ? player->openedMenu : true;

            // ---- ease the menu in / out
            const float target = menuOpen ? 1.0f : 0.0f;
            menuAnim += (target - menuAnim) * std::min(io.DeltaTime * 12.0f , 1.0f);
            if(std::abs(target - menuAnim) < 0.003f) menuAnim = target;

            // the mode of the view is the one of the Player ( the R key ) : it goes to the features
            if(player) data.features.currentMode = player->currentMode;

            drawHUD(data , io.Framerate , frameMs , menuOpen);

            if(menuAnim > 0.0f){
                // dim the scene a little behind the menu
                ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0.0f , 0.0f) , io.DisplaySize , IM_COL32(0 , 0 , 0 , static_cast<int>(95.0f * menuAnim)));

                drawMenu(data , io.Framerate , average , best , worst);
            }

            enforceLocks(data.features);
            rememberRasterChoice(data.features);
        }

        static const char* aoName(const uint32_t mode) noexcept{
            if(mode == HBAO_REQUIRED) return "HBAO";
            if(mode == SSAO_REQUIRED) return "SSAO";
            return "AO off";
        }

        static const char* indirectName(const uint32_t mode) noexcept{
            if(mode == SSGI_REQUIRED) return "SSGI";
            if(mode == SSR_REQUIRED) return "SSR";
            return "no indirect";
        }

        static const char* shadowName(const GeneralFeatures& f) noexcept{
            if(f.RayQuery) return "RayQuery shadows";
            if(f.VSM) return "VSM";
            return "no shadows";
        }

        static const char* modeName(const RenderMode mode) noexcept{
            if(mode == RenderMode::PathTracing) return "Path tracing";
            if(mode == RenderMode::RayTracing) return "Ray tracing";
            return "Raster";
        }

        // the small overlay that is always on the screen
        void drawHUD(const Data& data , const float fps , const float frameMs , const bool menuOpen) noexcept{
            ImGui::SetNextWindowPos(ImVec2(18.0f , 18.0f) , ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.42f);

            const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoMove;

            if(ImGui::Begin("##hud" , nullptr , flags)){
                const GeneralFeatures& f = data.features;
                const ImVec4 off = ImVec4(0.5f , 0.54f , 0.6f , 1.0f);

                ImGui::PushFont(nullptr , ImGui::GetStyle().FontSizeBase * 1.9f);
                ImGui::TextColored(fpsColor(fps) , "%.0f" , fps);
                ImGui::PopFont();
                ImGui::SameLine();
                ImGui::BeginGroup();
                ImGui::TextDisabled("FPS");
                ImGui::Text("%.2f ms" , frameMs);
                ImGui::EndGroup();

                chip(modeName(modeOf(f)) , ImVec4(1.0f , 0.72f , 0.30f , 1.0f));
                ImGui::SameLine(0.0f , 6.0f);
                chip(aoName(f.AO) , f.AO ? ImVec4(0.18f , 0.80f , 0.72f , 1.0f) : off);
                ImGui::SameLine(0.0f , 6.0f);
                chip(indirectName(f.indirectMode) , f.indirectMode ? ImVec4(0.62f , 0.50f , 1.0f , 1.0f) : off);
                ImGui::SameLine(0.0f , 6.0f);
                chip(shadowName(f) , (f.RayQuery || f.VSM) ? ImVec4(0.36f , 0.72f , 1.0f , 1.0f) : off);

                if(!menuOpen){
                    ImGui::TextDisabled("F  -  settings");
                }
            }
            ImGui::End();
        }

        void drawMenu(Data& data , const float fps , const float average , const float best , const float worst) noexcept{
            ImGuiIO& io = ImGui::GetIO();

            ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 520.0f - 24.0f , 24.0f) , ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(520.0f , 760.0f) , ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSizeConstraints(ImVec2(440.0f , 420.0f) , ImVec2(760.0f , io.DisplaySize.y - 24.0f));

            ImGui::PushStyleVar(ImGuiStyleVar_Alpha , menuAnim);

            const ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar;

            if(ImGui::Begin("settings##main" , nullptr , flags)){

                drawHeader();

                drawPerformance(fps , average , best , worst);

                drawPresets(data);

                // the sliders take the line without the room for the label
                ImGui::PushItemWidth(-250.0f);

                if(ImGui::BeginTabBar("##tabs")){

                    if(ImGui::BeginTabItem("Light")){
                        drawLight(data);
                        ImGui::EndTabItem();
                    }

                    if(ImGui::BeginTabItem("Features")){
                        drawFeatures(data);
                        ImGui::EndTabItem();
                    }

                    if(ImGui::BeginTabItem("Fog")){
                        drawFog(data);
                        ImGui::EndTabItem();
                    }

                    if(ImGui::BeginTabItem("Debug")){
                        drawDebug(data);
                        ImGui::EndTabItem();
                    }

                    ImGui::EndTabBar();
                }

                ImGui::PopItemWidth();
            }
            ImGui::End();

            ImGui::PopStyleVar();
        }

        // gradient strip with the title
        void drawHeader() noexcept{
            ImDrawList* list = ImGui::GetWindowDrawList();
            const ImVec2 position = ImGui::GetCursorScreenPos();
            const float width = ImGui::GetContentRegionAvail().x;
            const float height = 46.0f;
            const ImVec2 end(position.x + width , position.y + height);

            list->AddRectFilledMultiColor(position , end ,
                IM_COL32(18 , 150 , 138 , 255) , IM_COL32(110 , 82 , 230 , 255) ,
                IM_COL32(80 , 52 , 190 , 255) , IM_COL32(14 , 110 , 104 , 255));
            list->AddRect(position , end , IM_COL32(255 , 255 , 255 , 40) , 0.0f , 0 , 1.0f);

            ImGui::PushFont(nullptr , ImGui::GetStyle().FontSizeBase * 1.25f);
            const char* title = "RENDER SETTINGS";
            const ImVec2 titleSize = ImGui::CalcTextSize(title);
            list->AddText(ImVec2(position.x + 16.0f , position.y + (height - titleSize.y) * 0.5f) , IM_COL32(255 , 255 , 255 , 255) , title);
            ImGui::PopFont();

            const char* closeHint = "F to close";
            const ImVec2 hintSize = ImGui::CalcTextSize(closeHint);
            list->AddText(ImVec2(end.x - hintSize.x - 14.0f , position.y + (height - hintSize.y) * 0.5f) , IM_COL32(255 , 255 , 255 , 170) , closeHint);

            ImGui::Dummy(ImVec2(width , height));
        }

        void drawPerformance(const float fps , const float average , const float best , const float worst) noexcept{
            char overlay[64];
            std::snprintf(overlay , sizeof(overlay) , "avg %.2f ms   %.0f fps" , average , fps);

            // until the ring buffer is full only the written part is drawn ( it starts at index 0 )
            const bool full = frameFilled >= HISTORY;

            ImGui::PushStyleColor(ImGuiCol_PlotLines , fpsColor(fps));
            ImGui::PlotLines("##frames" , frameTimes.data() , full ? HISTORY : std::max(frameFilled , 1) , full ? frameOffset : 0 , overlay , 0.0f , graphMax , ImVec2(-1.0f , 64.0f));
            ImGui::PopStyleColor();

            ImGui::TextDisabled("best %.2f ms" , best < 1e8f ? best : 0.0f);
            ImGui::SameLine(0.0f , 16.0f);
            ImGui::TextDisabled("worst %.2f ms" , worst);
            if(image("finalImage")){
                ImGui::SameLine(0.0f , 16.0f);
                ImGui::TextDisabled("%ux%u" , image("finalImage")->extent.width , image("finalImage")->extent.height);
            }
        }

        // ------------------------------------------------------------------ presets

        struct Preset{
            const char* name;
            const char* hint;
            RenderMode mode;
            uint32_t ao;
            uint32_t aoIterations;
            float aoRadius;
            uint32_t indirectMode;
            uint32_t shadows;   // 0 = none , 1 = VSM , 2 = RayQuery
            uint32_t lightSamples;
            uint32_t sunSamples;
        };

        void applyPreset(const Preset& p , Data& data) noexcept{
            GeneralFeatures& f = data.features;

            // the raster choice of the preset first , then the mode takes what it locks
            rasterChoice.AO = p.ao;
            rasterChoice.indirectMode = p.indirectMode;
            rasterChoice.VSM = (p.shadows == 1) ? 1u : 0u;
            rasterChoice.RayQuery = (p.shadows == 2) ? 1u : 0u;

            f.AO_iteration = p.aoIterations;
            f.AOradius = p.aoRadius;
            f.RayQueryLightSampleCount = p.lightSamples;
            f.RayQuerySunSampleCount = p.sunSamples;

            applyMode(f , p.mode);
        }

        void drawPresets(Data& data) noexcept{
            static const Preset presets[4] = {
                {"Fast" , "Raster : SSAO with few samples , SSR , the shadows of the shadow maps"               , RenderMode::Raster      , SSAO_REQUIRED , 8u  , 0.5f , SSR_REQUIRED  , 1u , 0u , 0u},
                {"Quality"     , "Raster : HBAO , SSGI and the soft ray query shadows of the lamps and of the sun"      , RenderMode::Raster      , HBAO_REQUIRED , 48u , 1.2f , SSGI_REQUIRED , 2u , 4u , 4u},
                {"RT" , "Ray tracing : AO stays , the shadows and the screen space indirect are off"           , RenderMode::RayTracing  , HBAO_REQUIRED , 32u , 1.0f , NON_REQUIRED  , 0u , 0u , 0u},
                {"PT" , "Path tracing : only the path tracer , everything that it replaces is off"             , RenderMode::PathTracing , AO_NOT_REQUIRED , 16u , 1.0f , NON_REQUIRED , 0u , 0u , 0u}
            };

            ImGui::SeparatorText("Presets");

            const float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 3.0f) * 0.25f;
            for(int i = 0 ; i < 4 ; i++){
                if(i > 0) ImGui::SameLine();
                if(ImGui::Button(presets[i].name , ImVec2(width , 0.0f))){
                    applyPreset(presets[i] , data);
                }
                ImGui::SetItemTooltip("%s" , presets[i].hint);
            }
        }

        // ---------------------------------------------------------------- Light

        void drawLight(Data& data) noexcept{
            ImGui::SeparatorText("Light balance");

            // the indirect part and the direct part always make 1
            float indirect = data.general.indirectWeight;
            if(ImGui::SliderFloat("Indirect light proportion" , &indirect , 0.0f , 1.0f , "%.2f")){
                data.general.indirectWeight = indirect;
                data.general.directWeight = 1.0f - indirect;
                // the features have their own copy of the share
                data.features.IndirectShare = indirect;
            }
            helpMarker("How much of the picture is the indirect light ( bounces , reflections , the sky ) and how much is the direct one ( lamps , sun ).");
            ImGui::TextDisabled("Direct %.0f%%  /  indirect %.0f%%" , data.general.directWeight * 100.0f , data.general.indirectWeight * 100.0f);

            ImGui::SliderFloat("Exposure" , &data.general.exposure , 0.1f , 8.0f , "%.2f");

            bool normalMapOpenGL = data.general.normalMapOpenGL != 0u;
            if(ImGui::Checkbox("Normal maps are OpenGL style" , &normalMapOpenGL)){
                data.general.normalMapOpenGL = normalMapOpenGL ? 1u : 0u;
            }
            helpMarker("The green channel of the normal maps : OpenGL ( green up ) or DirectX ( green down ). If the light on the bumps looks inverted , flip it.");

            // the speed of the camera ( WASD / Space / Shift ) lives in the Player , not in a buffer : it is changed directly
            if(auto& user = inputData.context->user; user){
                ImGui::SliderFloat("Camera speed" , &user->speed , user->minSpeed , user->maxSpeed , "%.1f m/s");
            }

            // ===== sun =====
            ImGui::SeparatorText("Sun");

            sunPreview(sunElevation , sunAzimuth , sunColor , sunIntensity);

            float elevationDegrees = sunElevation * 57.2957795f;
            if(ImGui::SliderFloat("Elevation" , &elevationDegrees , -20.0f , 90.0f , "%.1f deg")){
                sunElevation = elevationDegrees / 57.2957795f;
                sunMoved = true;
            }
            helpMarker("The height of the sun above the horizon. Below 0 it is night.");

            float azimuthDegrees = sunAzimuth * 57.2957795f;
            if(azimuthDegrees < 0.0f) azimuthDegrees += 360.0f;
            if(ImGui::SliderFloat("Azimuth" , &azimuthDegrees , 0.0f , 360.0f , "%.1f deg")){
                sunAzimuth = azimuthDegrees / 57.2957795f;
                sunMoved = true;
            }
            helpMarker("The side of the sky where the sun is ( 0 is +X ). The sky is turned by it .");

            float color[3] = {sunColor.x , sunColor.y , sunColor.z};
            if(ImGui::ColorEdit3("Sun color" , color)){
                sunColor = glm::vec3(color[0] , color[1] , color[2]);
                sunMoved = true;
            }
            if(ImGui::SliderFloat("Sun intensity" , &sunIntensity , 0.0f , 40.0f , "%.2f" , ImGuiSliderFlags_Logarithmic)){
                sunMoved = true;
            }

            ImGui::TextDisabled("Zenith cos %.3f   azimuth cos %.3f / sin %.3f" , data.sky.sunCosZenith , data.sky.sunCosAzimuth , data.sky.sunSinAzimuth);
            ImGui::TextDisabled("Camera height %.1f m ( set by the application )" , data.sky.cameraHeight);

            if(ImGui::Button("Reset sun")){
                data.sun = defaultSun;
                sunFromData(defaultSun);
                sunMoved = false;
            }
            ImGui::SameLine();
            if(ImGui::Button("Reset balance")){
                data.general.indirectWeight = defaultIndirectWeight;
                data.general.directWeight = 1.0f - defaultIndirectWeight;
                data.features.IndirectShare = defaultIndirectWeight;
            }
        }

        // ---------------------------------------------------------------- Features

        // one radio button : it is greyed when the group is locked
        void drawFeatures(Data& data) noexcept{
            GeneralFeatures& f = data.features;

            // ===== the way of lighting =====
            ImGui::SeparatorText("Lighting mode");

            RenderMode mode = modeOf(f);
            int selected = static_cast<int>(mode);
            bool changed = false;
            changed |= ImGui::RadioButton("Raster" , &selected , static_cast<int>(RenderMode::Raster));
            ImGui::SameLine();
            changed |= ImGui::RadioButton("Ray tracing" , &selected , static_cast<int>(RenderMode::RayTracing));
            ImGui::SameLine();
            changed |= ImGui::RadioButton("Path tracing" , &selected , static_cast<int>(RenderMode::PathTracing));
            helpMarker("Ray tracing turns off the shadow maps , the ray query shadows , SSR and SSGI.\nPath tracing turns off all of them , AO too , and the ray tracing.\nWhat was chosen in the raster mode comes back when it is selected again.");
            if(changed){
                applyMode(f , static_cast<RenderMode>(selected));
            }

            const Locks locks = locksOf(modeOf(f));

            // ===== ambient occlusion =====
            ImGui::SeparatorText("Ambient occlusion");

            ImGui::BeginDisabled(locks.ao);
            {
                int ao = static_cast<int>(f.AO);
                bool aoChanged = false;
                aoChanged |= ImGui::RadioButton("Off##ao" , &ao , static_cast<int>(AO_NOT_REQUIRED));
                ImGui::SameLine();
                aoChanged |= ImGui::RadioButton("SSAO" , &ao , static_cast<int>(SSAO_REQUIRED));
                ImGui::SameLine();
                aoChanged |= ImGui::RadioButton("HBAO" , &ao , static_cast<int>(HBAO_REQUIRED));
                if(aoChanged) f.AO = static_cast<uint32_t>(ao);
                helpMarker("SSAO samples a hemisphere with a random kernel. HBAO marches the horizon in screen space : smoother and more expensive.");

                ImGui::BeginDisabled(f.AO == AO_NOT_REQUIRED);
                {
                    int iterations = static_cast<int>(f.AO_iteration);
                    if(ImGui::SliderInt("Samples" , &iterations , 1 , MAX_AO_ITERATION)){
                        f.AO_iteration = static_cast<uint32_t>(std::clamp(iterations , 1 , static_cast<int>(MAX_AO_ITERATION)));
                    }
                    ImGui::SliderFloat("Radius" , &f.AOradius , 0.05f , 5.0f , "%.2f");
                    ImGui::SliderFloat("Intensity" , &f.AO_intens , 0.0f , 4.0f , "%.2f");
                }
                ImGui::EndDisabled();
            }
            ImGui::EndDisabled();
            if(locks.ao) hint("Locked : the path tracer computes the occlusion itself.");

            // ===== shadows =====
            ImGui::SeparatorText("Shadows");

            ImGui::BeginDisabled(locks.shadows);
            {
                int shadows = f.RayQuery ? 2 : (f.VSM ? 1 : 0);
                bool shadowsChanged = false;
                shadowsChanged |= ImGui::RadioButton("Off##shadows" , &shadows , 0);
                ImGui::SameLine();
                shadowsChanged |= ImGui::RadioButton("VSM" , &shadows , 1);
                ImGui::SameLine();
                shadowsChanged |= ImGui::RadioButton("RayQuery" , &shadows , 2);
                // one source of the shadows at a time
                if(shadowsChanged){
                    f.VSM = (shadows == 1) ? 1u : 0u;
                    f.RayQuery = (shadows == 2) ? 1u : 0u;
                }
                helpMarker("VSM : virtual shadow maps. RayQuery : the rays of the lamp and of the sun in the cone of their size , soft with the distance.");

                ImGui::BeginDisabled(!f.RayQuery);
                {
                    int lightSamples = static_cast<int>(f.RayQueryLightSampleCount);
                    if(ImGui::SliderInt("Lamp rays" , &lightSamples , 0 , 16)){
                        f.RayQueryLightSampleCount = static_cast<uint32_t>(std::clamp(lightSamples , 0 , 16));
                    }
                    helpMarker("Rays to the lamp in the cone of its size. 0 : the lamp casts no shadow.");

                    int sunSamples = static_cast<int>(f.RayQuerySunSampleCount);
                    if(ImGui::SliderInt("Sun rays" , &sunSamples , 0 , 16)){
                        f.RayQuerySunSampleCount = static_cast<uint32_t>(std::clamp(sunSamples , 0 , 16));
                    }
                    helpMarker("Rays to the sun in the cone of its disc. 0 : the sun casts no shadow.");

                    ImGui::SliderFloat("Lamp / sun proportion" , &f.light_sun_Proportion , 0.0f , 1.0f , "%.2f");
                    helpMarker("0 : only the shadow of the lamp , 1 : only the shadow of the sun.");
                    ImGui::SliderFloat("Temporal accumulation" , &f.temporalShadowK , 0.0f , 0.98f , "%.2f");
                }
                ImGui::EndDisabled();

                ImGui::SliderFloat("Shadow thickness" , &f.shadowThickness , 0.1f , 4.0f , "%.2f");
            }
            ImGui::EndDisabled();
            if(locks.shadows) hint("Locked : the traced light has its own shadows.");

            // ===== indirect light =====
            ImGui::SeparatorText("Indirect light");

            ImGui::BeginDisabled(locks.indirect);
            {
                int indirect = static_cast<int>(f.indirectMode);
                bool indirectChanged = false;
                indirectChanged |= ImGui::RadioButton("None" , &indirect , static_cast<int>(NON_REQUIRED));
                ImGui::SameLine();
                indirectChanged |= ImGui::RadioButton("SSR" , &indirect , static_cast<int>(SSR_REQUIRED));
                ImGui::SameLine();
                indirectChanged |= ImGui::RadioButton("SSGI" , &indirect , static_cast<int>(SSGI_REQUIRED));
                if(indirectChanged) f.indirectMode = static_cast<uint32_t>(indirect);
                helpMarker("SSR : screen space reflections. SSGI : the screen space diffuse bounce. One of them at a time.");

                ImGui::SliderFloat("Max distance" , &f.maxDepth , 1.0f , 200.0f , "%.0f m");
            }
            ImGui::EndDisabled();
            if(locks.indirect) hint("Locked : the traced light has its own reflections and bounces.");

            // ===== the rest =====
            ImGui::SeparatorText("Other");

            bool fog = f.volumentricFog != 0;
            if(ImGui::Checkbox("Volumetric fog" , &fog)){
                f.volumentricFog = fog ? 1u : 0u;
            }
            helpMarker("The fog volume is lit by the sun and the lamps. The parameters are in the Fog tab.");

            ImGui::Spacing();
            if(ImGui::Button("Reset features")){
                f = defaultFeatures;
                rasterChoice.AO = f.AO;
                rasterChoice.indirectMode = f.indirectMode;
                rasterChoice.VSM = f.VSM;
                rasterChoice.RayQuery = f.RayQuery;
                enforceLocks(f);
            }
        }

        // ---------------------------------------------------------------- Fog

        void drawFog(Data& data) noexcept{
            FogVolume& fog = data.fog;

            ImGui::SeparatorText("Medium");

            ImGui::SliderFloat("Density" , &fog.density , 0.0f , 0.5f , "%.4f" , ImGuiSliderFlags_Logarithmic);
            helpMarker("The extinction in 1/m : how fast the fog hides the things that are behind it.");

            float albedo[3] = {fog.albedo.x , fog.albedo.y , fog.albedo.z};
            if(ImGui::ColorEdit3("Albedo" , albedo)){
                fog.albedo = glm::vec3(albedo[0] , albedo[1] , albedo[2]);
            }
            helpMarker("The color that the fog gives to the light of the lamps and of the sun.");

            // the own glow : visible without any light , it is the color of the fog far away
            float glow[3] = {fogColor.x , fogColor.y , fogColor.z};
            if(ImGui::ColorEdit3("Fog color" , glow)){
                fogColor = glm::vec3(glow[0] , glow[1] , glow[2]);
                fog.color = fogColor * fogIntensity;
            }
            if(ImGui::SliderFloat("Fog glow" , &fogIntensity , 0.0f , 4.0f , "%.3f" , ImGuiSliderFlags_Logarithmic)){
                fog.color = fogColor * fogIntensity;
            }
            helpMarker("The own radiance of the fog : the fog is of this color even where no light reaches it ( 0 : the fog is only what the lights light ).");

            ImGui::SliderFloat("Anisotropy ( g )" , &fog.g , -0.95f , 0.95f , "%.2f");
            helpMarker("The phase function of the scattering. 0 : the same in all directions. Above 0 : the fog glows around the lights ( forward scattering ).");

            ImGui::SeparatorText("Height");

            ImGui::SliderFloat("Height falloff" , &fog.heightFallOff , 0.0f , 1.0f , "%.3f" , ImGuiSliderFlags_Logarithmic);
            helpMarker("How fast the density goes down above the base height ( 1/m ). 0 : the fog is the same at every height.");
            ImGui::DragFloat("Base height" , &fog.fogBaseHeight , 0.1f , -100.0f , 200.0f , "%.1f m");

            ImGui::SeparatorText("Volume");

            float center[3] = {fog.center.x , fog.center.y , fog.center.z};
            if(ImGui::DragFloat3("Center" , center , 0.1f)){
                fog.center = glm::vec3(center[0] , center[1] , center[2]);
            }

            float extent[3] = {fog.halfExtent.x , fog.halfExtent.y , fog.halfExtent.z};
            if(ImGui::DragFloat3("Half extent" , extent , 0.1f , 0.1f , 1000.0f)){
                fog.halfExtent = glm::vec3(extent[0] , extent[1] , extent[2]);
            }

            ImGui::SliderFloat("Feather" , &fog.feather , 0.0f , 50.0f , "%.1f m");
            helpMarker("The width of the soft edge of the box of the fog.");

            ImGui::TextDisabled("Depth slices : %.0f ( set by the size of the fog image )" , fog.gridSize);

            if(ImGui::Button("Reset fog")){
                fog = defaultFog;
                fogFromData(defaultFog);
            }
        }

        // ---------------------------------------------------------------- Debug

        void drawDebug(Data& data) noexcept{
            auto& player = inputData.context->user;
            GeneralFeatures& f = data.features;

            ImGui::SeparatorText("View");

            static const char* modes[TOTAL_VIEW_MODES] = {"Final image" , "Albedo" , "Normal" , "Depth ( the farther , the redder )" , "Metallic" , "Roughness" , "Indirect light only" , "Ambient occlusion" , "Shadow ( 1 = lit , red = more than 1 )"};
            int mode = static_cast<int>(std::min(f.currentMode , static_cast<uint32_t>(TOTAL_VIEW_MODES - 1)));
            if(ImGui::Combo("Output" , &mode , modes , TOTAL_VIEW_MODES)){
                f.currentMode = static_cast<uint32_t>(mode);
                // the R key cycles Player::currentMode , keep both in sync
                if(player) player->currentMode = static_cast<uint32_t>(mode);
            }

            ImGui::SeparatorText("Active flags");

            if(ImGui::BeginTable("##flags" , 4 , ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)){
                auto flag = [&](const char* name , const uint32_t value){
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%s" , name);
                    ImGui::SameLine();
                    ImGui::TextColored(value ? ImVec4(0.30f , 0.90f , 0.50f , 1.0f) : ImVec4(0.55f , 0.58f , 0.64f , 1.0f) , "%u" , value);
                };

                flag("AO" , f.AO);
                flag("Indirect" , f.indirectMode);
                flag("VSM" , f.VSM);
                flag("RayQuery" , f.RayQuery);
                flag("RT" , f.RT);
                flag("PT" , f.PT);
                flag("Fog" , f.volumentricFog);
                flag("Mode" , f.currentMode);

                ImGui::EndTable();
            }

            ImGui::Spacing();

            if(ImGui::Button("Copy settings")){
                const std::string text = std::format(
                    "mode={} AO={} samples={} radius={:.2f}\nindirect={} VSM={} RayQuery={} lampRays={} sunRays={}\nsun elevation={:.1f} azimuth={:.1f} intensity={:.2f}\nfog density={:.4f} g={:.2f}\n" ,
                    modeName(modeOf(f)) , f.AO , f.AO_iteration , f.AOradius , f.indirectMode , f.VSM , f.RayQuery , f.RayQueryLightSampleCount , f.RayQuerySunSampleCount ,
                    sunElevation * 57.2957795f , sunAzimuth * 57.2957795f , sunIntensity , data.fog.density , data.fog.g);
                ImGui::SetClipboardText(text.c_str());
            }
        }
    };

    using pImGuiPipe = std::shared_ptr<ImGuiPipe>;
}
