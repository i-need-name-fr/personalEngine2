#pragma once

#include "include/Buffer2/GeneralBuffer.hpp"
#include "include/Buffer2/SourceBuffers.hpp"
#include "include/Image2/ViewPerLayer.hpp"
#include "include/Image2/ViewPerMip.hpp"
#include "include/Image2/FrameTargets.hpp"

#include "include/AS2/TLAS2.hpp"
#include "include/Object2/cube.hpp"
#include "include/Image2/MultiViewImage2.hpp"

#include "include/generalPipelines2/deferredRender.hpp"
#include "include/generalPipelines2/TransmissionLUT.hpp"
#include "include/generalPipelines2/MultiScatterLUT.hpp"
#include "include/generalPipelines2/MultiScatter.hpp"
#include "include/generalPipelines2/AtmosphereScattering.hpp"
#include "include/generalPipelines2/ForwardPlus.hpp"
#include "include/generalPipelines2/AmbientOcclusion.hpp"
#include "include/generalPipelines2/RayQueryShadows.hpp"
#include "include/generalPipelines2/BIlateralFIltering.hpp"
#include "include/generalPipelines2/VolumentricRender.hpp"
#include "include/generalPipelines2/LightPass.hpp"
#include "include/generalPipelines2/IndirectLightSum.hpp"
#include "include/generalPipelines2/ReBLUR.hpp"
#include "include/generalPipelines2/ReLAX.hpp"
#include "include/generalPipelines2/PostProcess.hpp"
#include "include/generalPipelines2/AEHDR.hpp"
#include "include/generalPipelines2/HDR.hpp"
#include "include/generalPipelines2/imgui.hpp"

#include "include/GraphRender2/GraphRender2.hpp"

namespace mox{

    // the test scene : a grass floor , the old cubes , four lamps ( two of them orbit like Phobos and Deimos ) and the whole chain of the passes
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
                frameTime = dt;

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

        // ---------------------------------------------------------------- the scene

        // the top of the floor
        static constexpr float FLOOR_Y = -2.0f;
        static constexpr uint32_t LAMP_COUNT = 4;

        struct Lamp{
            glm::vec3 color;
            float intensity;
            float radius;
        };

        // the lamps : 0 - Phobos ( close , fast ) , 1 - Deimos ( far , slow , the other way ) , 2 and 3 stand still
        inline static const Lamp lampInfo[LAMP_COUNT] = {
            {{1.00f , 0.55f , 0.25f} , 14.0f , 30.0f},
            {{0.30f , 0.60f , 1.00f} , 20.0f , 40.0f},
            {{1.00f , 0.90f , 0.70f} , 12.0f , 30.0f},
            {{0.85f , 0.30f , 1.00f} , 12.0f , 30.0f},
        };

        // the position of a lamp at the time t ( seconds )
        static glm::vec3 lampPosition(const uint32_t index , const float t) noexcept{
            switch(index){
                case 0:{
                    // Phobos : a close and fast orbit around the middle of the scene , it bobs a little
                    const float a = t * 1.2f;
                    return glm::vec3(std::cos(a) * 8.0f , 2.0f + 0.8f * std::sin(a * 2.0f) , std::sin(a) * 8.0f);
                }
                case 1:{
                    // Deimos : a far and slow orbit the other way round , the plane of the orbit is tilted
                    const float a = -t * 0.45f + 3.14159265f;
                    const float tilt = glm::radians(20.0f);
                    return glm::vec3(std::cos(a) * 13.0f , 5.0f + std::sin(a) * std::sin(tilt) * 13.0f , std::sin(a) * std::cos(tilt) * 13.0f);
                }
                case 2: return glm::vec3(-10.0f , 1.0f , -4.0f);
                default: return glm::vec3(10.0f , 2.5f , -9.0f);
            }
        }

        // the matrix of the body of a lamp
        static glm::mat4 lampMatrix(const glm::vec3 position) noexcept{
            return glm::scale(glm::translate(glm::mat4(1.0f) , position) , glm::vec3(0.3f));
        }

        // sets the matrix of an instance ( a unit cube ) and the box that culls it
        static void setInstance(const uint32_t instanceID , const glm::mat4& matrix) noexcept{
            if(instanceID >= Mesh2::instances.size()) return;
            Mesh2::instances[instanceID] = matrix;

            for(auto& mesh : Mesh2::totalMeshesData){
                if(mesh.instanceIndex != instanceID) continue;

                glm::vec3 boxMin(std::numeric_limits<float>::max());
                glm::vec3 boxMax(std::numeric_limits<float>::lowest());
                for(uint32_t corner = 0 ; corner < 8 ; corner++){
                    const glm::vec3 local((corner & 1u) ? 1.0f : -1.0f , (corner & 2u) ? 1.0f : -1.0f , (corner & 4u) ? 1.0f : -1.0f);
                    const glm::vec3 world = glm::vec3(matrix * glm::vec4(local , 1.0f));
                    boxMin = glm::min(boxMin , world);
                    boxMax = glm::max(boxMax , world);
                }
                mesh.boxMin = boxMin;
                mesh.boxMax = boxMax;
                mesh.boxCenter = (boxMin + boxMax) * 0.5f;
                mesh.boxRadius = glm::length(boxMax - mesh.boxCenter);
            }
        }

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

                // the camera starts in front of the scene
                context->user->position = glm::vec3(0.0f , 3.0f , 17.0f);
                // the direction is made from yaw / pitch by the Player ( calcFront ) , so these are the ones that are set
                context->user->yaw = -90.0f;
                context->user->pitch = -10.0f;
                context->user->playerMoved = true;
                context->user->calcFront();

                // ---------------------------------------------------------------- the meshes
                {
                    const auto result = Mesh2::Begin(context.get() , manager.get());
                    if(!result.has_value()) return result;
                }

                {
                    // the old cubes ( the first instance of a mesh is made by the constructor itself , at the origin )
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
                }

                {
                    // the floor : a flat box with the grass , the texture is repeated ( the UV of the cube are multiplied )
                    std::vector<Vertex> vertices = cube::vertices;
                    for(auto& vertex : vertices) vertex.UV *= 24.0f;

                    Mesh2::CreateInfo info{};
                    info.color = glm::vec3(1.0f);
                    info.diffusePath = "../textures/grassDiffuse.png";
                    info.depthPath = "../textures/grassDepth.png";
                    info.roughnessPath = "../textures/grassRoughness.png";
                    info.normalPath = "../textures/grassNormal.png";
                    info.needsCalcTBN = true;
                    info.inputVertices = vertices;
                    info.inputIndices = cube::indices;

                    info.roughness = 0.9f;
                    info.metallic = 0.0f;
                    info.isALightObject = false;

                    floorInstance = static_cast<uint32_t>(Mesh2::instances.size());
                    floorMesh = std::make_shared<Mesh2>(info);

                    // the first instance of the mesh is the floor itself
                    glm::mat4 model = glm::translate(glm::mat4(1.0f) , glm::vec3(0.0f , FLOOR_Y - 0.5f , 0.0f));
                    model = glm::scale(model , glm::vec3(45.0f , 0.5f , 45.0f));
                    setInstance(floorInstance , model);
                }

                // the bodies of the lamps : small cubes that glow with the color of the lamp ( a mesh has one emission , so one mesh per lamp )
                for(uint32_t i = 0 ; i < LAMP_COUNT ; i++){
                    Mesh2::CreateInfo info{};
                    info.color = glm::vec3(0.1f);
                    info.emission = lampInfo[i].color * 6.0f;
                    info.needsCalcTBN = true;
                    info.inputVertices = cube::vertices;
                    info.inputIndices = cube::indices;

                    info.roughness = 0.5f;
                    info.metallic = 0.0f;
                    info.isALightObject = true;

                    lampInstance[i] = static_cast<uint32_t>(Mesh2::instances.size());
                    lampMesh[i] = std::make_shared<Mesh2>(info);
                    // the first instance of the mesh is the body itself
                    setInstance(lampInstance[i] , lampMatrix(lampPosition(i , 0.0f)));
                }

                {
                    const auto result = Mesh2::End();
                    if(!result.has_value()) return result;
                }

                // ---------------------------------------------------------------- the buffers
                sbuf = std::make_shared<SourceBuffers>(context.get());
                tlas = std::make_shared<TLAS2>(context.get());

                // the values that the UI shows when it starts ( it takes them from the buffers , so they are written before the UI is made )
                initBufferData();

                graph = std::make_shared<GraphRender2>();

                // ---------------------------------------------------------------- the LUTs ( made once )
                {
                    TransmissionLUTPipe::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/atmosphereScattering/transmissionLUT.spv";
                    info.image = source->transmissionLUT.get();

                    transmissionPipe = std::make_shared<TransmissionLUTPipe>(info);
                }

                {
                    MultiScatterLUTPipe::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/atmosphereScattering/multiScattering.spv";
                    info.transmission = source->transmissionLUT.get();
                    info.image = source->multiScatterLUT.get();

                    multiScatterLUTPipe = std::make_shared<MultiScatterLUTPipe>(info);
                }

                {
                    MultiScatter::CreateInfo info{};
                    info.context = context.get();
                    info.image = source->multiScatter.get();
                    info.path = "../Shaders/MultiScatter/MultiScatter.spv";

                    multiScatter = std::make_shared<MultiScatter>(info);
                }

                // ---------------------------------------------------------------- the passes
                {
                    AtmosphereScattering::CreateInfo info{};
                    info.context = context.get();
                    info.skyView = "../Shaders/atmosphereScattering/SkyViewLUT.spv";
                    info.transition = "../Shaders/atmosphereScattering/transitionCube.spv";
                    atmosphere = std::make_shared<AtmosphereScattering>(info);
                }

                {
                    DeferredRender::CreateInfo info{};
                    info.context = context.get();
                    info.fragment = "../Shaders/DeferredRender/fragmentShader.spv";
                    info.vertex = "../Shaders/DeferredRender/vertexShader.spv";
                    info.manager = manager.get();

                    render = std::make_shared<DeferredRender>(info);
                }

                {
                    ForwardPlus::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/ForwardPlus/forward.spv";
                    info.generalData = forwardData;

                    forward = std::make_shared<ForwardPlus>(info);
                }

                {
                    AmbientOcclusion::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/AmbientOcclusion/AmbientOcclusion.spv";

                    ao = std::make_shared<AmbientOcclusion>(info);
                }

                {
                    RayQueryShadows::CreateInfo info{};
                    info.context = context.get();
                    info.as = tlas.get();
                    info.path = "../Shaders/RayQueryShadows/rayQueryShadows.spv";

                    shadows = std::make_shared<RayQueryShadows>(info);
                }

                {
                    BilateralFiltering::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/BF_singular/bilateralFilteringSingular.spv";

                    bilateral = std::make_shared<BilateralFiltering>(info);
                }

                {
                    VolumentricRender::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/volumentricRender/fogCalc.spv";

                    fog = std::make_shared<VolumentricRender>(info);
                }

                {
                    LightPass::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/LightPass/lightPass.spv";

                    lightPass = std::make_shared<LightPass>(info);
                }

                {
                    IndirectLightSum::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/indirectShader/indirectCalc.spv";

                    indirect = std::make_shared<IndirectLightSum>(info);
                }

                {
                    ReBLUR::CreateInfo info{};
                    info.context = context.get();
                    info.spatial = "../Shaders/denoiser/ReBLUR_spatial.spv";
                    info.temporal = "../Shaders/denoiser/ReBLUR_temporal.spv";

                    reblur = std::make_shared<ReBLUR>(info);
                }

                {
                    ReLAX::CreateInfo info{};
                    info.context = context.get();
                    info.spatial = "../Shaders/denoiser/ReLAX_spatial.spv";
                    info.temporal = "../Shaders/denoiser/ReLAX_temporal.spv";

                    relax = std::make_shared<ReLAX>(info);
                }

                {
                    PostProcess::CreateInfo info{};
                    info.context = context.get();
                    info.path = "../Shaders/PostProcess/PostProcess.spv";

                    post = std::make_shared<PostProcess>(info);
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

                {
                    // the pointers are DeviceReadableBuffer* : the UI reads the CPU copy of the buffers and writes the changes back
                    ImGuiPipe::CreateInfo info{};
                    info.context = context.get();
                    info.generalData = static_cast<void*>(sbuf->generalBuffer.get());
                    info.features = static_cast<void*>(sbuf->features.get());
                    info.fogData = static_cast<void*>(sbuf->fogData.get());
                    info.sunData = static_cast<void*>(sbuf->sunBuffer.get());
                    info.skyViewData = static_cast<void*>(sbuf->skyBuffer.get());

                    imgui = std::make_shared<ImGuiPipe>(info);
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
                const auto result = graph->processInputParameter(sbuf.get());
                if(!result.has_value()) return result;
            }

            // the order of the frame
            const auto add = [this](GraphPass2* pass) -> ErrorDataOutput<void>{
                return graph->addGraphPass2(pass , "main");
            };

            GraphPass2* order[] = {
                atmosphere.get() ,   // the sky : the sky view LUT and the cube
                render.get() ,       // the g buffer
                forward.get() ,      // the lamps of every tile
                ao.get() ,           // the occlusion
                shadows.get() ,      // the shadows of the lamps and of the sun
                bilateral.get() ,    // the filter of the shadows and of the occlusion
                fog.get() ,          // the froxel volume of the fog
                lightPass.get() ,    // the direct light and the sky
                indirect.get() ,     // the screen space bounces and reflections
                reblur.get() ,       // diffuse
                relax.get() ,        // specular
                post.get() ,         // diffuse * albedo + specular , the fog
                aehdr.get() ,        // the exposure
                hdr.get() ,          // the tone mapping into the final image
                imgui.get() ,        // the UI is the last one
            };

            for(GraphPass2* pass : order){
                const auto result = add(pass);
                if(!result.has_value()) return result;
            }

            return {};
        }

        void processDelete() noexcept {
            if(context && context->logicalDevice) vkDeviceWaitIdle(context->logicalDevice->device);

            // the graph keeps raw pointers to the passes , and the passes to the resources , so they go first
            graph = nullptr;
            imgui = nullptr;
            hdr = nullptr;
            aehdr = nullptr;
            post = nullptr;
            relax = nullptr;
            reblur = nullptr;
            indirect = nullptr;
            lightPass = nullptr;
            fog = nullptr;
            bilateral = nullptr;
            shadows = nullptr;
            ao = nullptr;
            forward = nullptr;
            render = nullptr;
            atmosphere = nullptr;
            multiScatter = nullptr;
            multiScatterLUTPipe = nullptr;
            transmissionPipe = nullptr;

            tlas = nullptr;
            sbuf = nullptr;
            mesh = nullptr;
            floorMesh = nullptr;
            for(auto& lamp : lampMesh) lamp = nullptr;
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

        // ---------------------------------------------------------------- the data of the buffers

        // the buffers are HOST_VISIBLE | HOST_COHERENT : one write of the CPU copy is what the shaders read . ( a vkCmdUpdateBuffer on top of it would be executed later ,
        // when the GPU gets to this frame , and it would put the old value over what the UI wrote in a newer frame )
        template<typename T>
        void writeBuffer(DeviceReadableBuffer* buffer , T& value) noexcept{
            buffer->recordData(std::span<T>(&value , 1));
        }

        // the first values of the buffers that the UI owns ( the CPU copy only : the first frame sends them to the GPU )
        void initBufferData() noexcept{
            {
                GeneralInfo info{};
                info.exposure = 1.6f;
                // the same share that the UI starts with
                info.directWeight = 0.5f;
                info.indirectWeight = 0.5f;
                info.ambientInIndirect = 1;
                sbuf->generalBuffer->recordData(std::span<GeneralInfo>(&info , 1));
            }

            {
                GeneralFeatures features{};
                features.AO = SSAO_REQUIRED;
                features.AO_iteration = 16;
                features.AOradius = 0.6f;
                features.AO_intens = 1.0f;
                features.indirectMode = SSGI_REQUIRED;
                features.VSM = 0;
                features.RayQuery = 1;
                features.RayQueryLightSampleCount = 4;
                features.RayQuerySunSampleCount = 4;
                features.shadowThickness = 1.0f;
                features.temporalShadowK = 0.85f;
                features.light_sun_Proportion = 0.5f;
                features.RT = 0;
                features.PT = 0;
                features.currentMode = DEFAULT_MODE;
                features.randomValues = 1;
                features.volumentricFog = 1;
                features.IndirectShare = 0.5f;
                features.maxDepth = 50.0f;
                sbuf->features->recordData(std::span<GeneralFeatures>(&features , 1));
            }

            {
                // a thin haze : the lamps get a glow in it , the horizon fades into the color of the fog
                FogVolume fogVolume{};
                fogVolume.center = glm::vec3(0.0f);
                fogVolume.halfExtent = glm::vec3(1000.0f);
                fogVolume.albedo = glm::vec3(1.0f);
                fogVolume.density = 0.012f;
                fogVolume.feather = 0.0f;
                fogVolume.g = 0.45f;
                fogVolume.gridSize = 64.0f;
                fogVolume.heightFallOff = 0.12f;
                fogVolume.fogBaseHeight = FLOOR_Y;
                fogVolume.color = glm::vec3(0.35f , 0.45f , 0.6f) * 0.25f;
                sbuf->fogData->recordData(std::span<FogVolume>(&fogVolume , 1));
            }

            {
                // the sun : 40 degrees above the horizon , the light travels away from the sun
                const float elevation = glm::radians(40.0f);
                const float azimuth = glm::radians(215.0f);
                const glm::vec3 toSun(std::cos(elevation) * std::cos(azimuth) , std::sin(elevation) , std::cos(elevation) * std::sin(azimuth));

                SunSource sun{};
                sun.direction = -toSun;
                sun.diffuse = glm::vec3(1.0f , 0.95f , 0.85f) * 4.0f;
                sbuf->sunBuffer->recordData(std::span<SunSource>(&sun , 1));

                // the angles of the sky view data : the same as the UI derives from the sun
                const float horizontal = std::sqrt(toSun.x * toSun.x + toSun.z * toSun.z);
                SkyViewData sky{};
                sky.cameraHeight = 5.0f;
                sky.sunCosZenith = toSun.y;
                sky.sunCosAzimuth = toSun.x / horizontal;
                sky.sunSinAzimuth = toSun.z / horizontal;
                sbuf->skyBuffer->recordData(std::span<SkyViewData>(&sky , 1));
            }

            {
                forwardData.TILE_SIZE = 16;
                forwardData.TILE_PER_ROW = (context->sc->extent.width + forwardData.TILE_SIZE - 1) / forwardData.TILE_SIZE;
                forwardData.TILE_TOTAL_AMOUNT = forwardData.TILE_PER_ROW * ((context->sc->extent.height + forwardData.TILE_SIZE - 1) / forwardData.TILE_SIZE);
                forwardData.totalLights = LAMP_COUNT;
                sbuf->forwardInfo->recordData(std::span<ForwardPlusInfo>(&forwardData , 1));
            }
        }

        // update data
        void updateData(VkCommandBuffer &cmd) noexcept{

            // the buffers are written by the CPU right now ( HOST_VISIBLE memory ) , and the context lets the CPU run 3 frames ahead of the GPU ( frames_in_flight ) :
            // without waiting , the GPU that still draws the previous frame reads the matrices , the frame index and the prevViewProj of THIS frame in the middle of its work .
            // the motion vectors then do not match the g buffer and the denoisers smear the picture ( a floor that turns into soap when standing still , a picture that drifts when
            // moving ) , and it differs from run to run . so the GPU is waited here : the CPU and the GPU do not overlap , and the frame is always consistent .
            // ( the way to keep the overlap : one copy of these buffers per frame in flight )
            vkQueueWaitIdle(context->graphicsQueue->queue);

            const float time = static_cast<float>(glfwGetTime());

            const float aspect = static_cast<float>(context->sc->extent.width) / static_cast<float>(context->sc->extent.height);
            glm::mat4 proj = glm::perspective(glm::radians(70.0f) , aspect , 0.001f , 200.0f);
            glm::mat4 view = glm::lookAt(context->user->position , context->user->position + context->user->direction , glm::vec3(0.0f , 1.0f , 0.0f));

            // the buffers belong to the UI too : the CPU copy is taken and only the fields of the application are changed
            {
                GeneralInfo info = sbuf->generalBuffer->getBuffetDataSingular<GeneralInfo>();
                // the frame counter is increased by the context at the end of the frame , the passes ping-pong by it
                info.frameIndex = context->sc->frameIndex;
                info.dt = frameTime;
                info.viewOnly = view;
                info.projOnly = proj;
                info.viewMode = context->user->currentMode;
                info.inverseView = glm::inverse(view);
                info.inverseProj = glm::inverse(proj);
                info.invViewProj = glm::inverse(proj * view);
                info.viewProj = proj * view;
                info.viewSkybox = glm::mat4(glm::mat3(view));
                if(firstIteration){
                    firstIteration = false;
                    info.prevViewProj = info.viewProj;
                }else{
                    info.prevViewProj = prevVP;
                }

                info.lightCount = LAMP_COUNT;
                info.playerDirection = context->user->direction;
                info.playerFar = 200.0f;
                info.playerNear = 0.001f;
                info.screenHeight = context->sc->extent.height;
                info.screenWidth = context->sc->extent.width;
                info.playerPosition = context->user->position;

                prevVP = proj * view;

                writeBuffer(sbuf->generalBuffer.get() , info);
            }

            // the features , the fog and the sun are the UI's : they are only sent to the GPU ( nothing is rewritten )
            {
                GeneralFeatures features = sbuf->features->getBuffetDataSingular<GeneralFeatures>();
                writeBuffer(sbuf->features.get() , features);

                FogVolume fogVolume = sbuf->fogData->getBuffetDataSingular<FogVolume>();
                writeBuffer(sbuf->fogData.get() , fogVolume);

                SunSource sun = sbuf->sunBuffer->getBuffetDataSingular<SunSource>();
                writeBuffer(sbuf->sunBuffer.get() , sun);

                // the height of the camera above the ground is the only field of the sky view data that the application owns
                SkyViewData sky = sbuf->skyBuffer->getBuffetDataSingular<SkyViewData>();
                sky.cameraHeight = std::max(context->user->position.y - FLOOR_Y , 0.5f);
                writeBuffer(sbuf->skyBuffer.get() , sky);

                writeBuffer(sbuf->forwardInfo.get() , forwardData);
            }

            // the lamps : the light and the body of every one of them
            {
                std::array<LightSource , LAMP_COUNT> lights{};
                for(uint32_t i = 0 ; i < LAMP_COUNT ; i++){
                    const glm::vec3 position = lampPosition(i , time);

                    lights[i].diffuse = lampInfo[i].color * lampInfo[i].intensity;
                    lights[i].position = position;
                    lights[i].lightLinear = 0.09f;
                    lights[i].lightQuadratic = 0.032f;
                    lights[i].lightRadius = lampInfo[i].radius;

                    // the static ones were placed once
                    if(i < 2) setInstance(lampInstance[i] , lampMatrix(position));
                }

                sbuf->lightBuffer->recordData(std::span<LightSource>(lights));

                // the moved instances
                Mesh2::instanceBuffer->recordData(translateToSpan(Mesh2::instances));
                Mesh2::MeshBuffer->recordData(translateToSpan(Mesh2::totalMeshesData));

                // the TLAS follows the lamps : the update is made here , the ray query shadows read it later
                tlas->updateTLAS(cmd);

                VkMemoryBarrier2 barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT;

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.memoryBarrierCount = 1;
                info.pMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd , &info);
            }
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
        std::shared_ptr<SourceBuffers> sbuf{}; // created
        pTLAS2 tlas{}; // created
        pTextureManager2 manager; // created

        pMesh2 mesh; // created
        pMesh2 floorMesh; // created
        pMesh2 lampMesh[LAMP_COUNT]; // created
        uint32_t floorInstance{0};
        uint32_t lampInstance[LAMP_COUNT]{};
        std::vector<VkImage> images{}; // created
        std::vector<VkImageView> views{}; // created

        // made once , they only fill an image
        pTransmissionLUTPipe transmissionPipe{};
        pMultiScatterLUTPipe multiScatterLUTPipe{};
        pMultiScatter multiScatter{};

        pAtmosphereScattering atmosphere{};
        pDeferredRender render{};
        pForwardPlus forward{};
        pAmbientOcclusion ao{};
        pRayQueryShadows shadows{};
        pBilateralFiltering bilateral{};
        pVolumentricRender fog{};
        pLightPass lightPass{};
        pIndirectLightSum indirect{};
        pReBLUR reblur{};
        pReLAX relax{};
        pPostProcess post{};
        pAEHDR aehdr{};
        pHDR  hdr{};
        pImGuiPipe imgui{};

        ForwardPlusInfo forwardData{};

        glm::mat4 prevVP{1.0f};
        bool firstIteration{true};
        float frameTime{0.016f};

        QueueRecordTypeFunction recordFunction{};

        pGraphRender2 graph{};
    };

}
