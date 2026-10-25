#pragma once

#include "include/Buffer2/GeneralBuffer.hpp"
#include "include/Buffer2/SourceBuffers.hpp"
#include "include/Image2/ViewPerLayer.hpp"
#include "include/Image2/ViewPerMip.hpp"
#include "include/Image2/FrameTargets.hpp"
#include "include/Object2/Model2.hpp"
#include "include/Object2/Model3.hpp"
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
#include "include/generalPipelines2/imgui2.hpp"

#include "include/GraphRender2/GraphRender2.hpp"
#include "include/generalPipelines2/RayTracing.hpp"
#include "include/generalPipelines2/PathTracing.hpp"
#include "include/RTCompileObjects/RTCompileObjects.hpp"
#include "include/LightManager/LightManager.hpp"
#include "include/generalPipelines2/RadianceCascade.hpp"

namespace gui = ImGui;

namespace mox{

    inline constexpr uint32_t RayTracingRequired = 2;
    inline constexpr uint32_t PathTracingRequired = 3;
    inline constexpr uint32_t RasterizationRequired = 1;
    inline constexpr std::string RasterizationPath = "4356456sdf";
    inline constexpr std::string RayTracingPath = "q3456wera";
    inline constexpr std::string PathTracingPath = "f34q5rtggw45";


    class Core2 final{
    public:

        SetTypicalOperators(Core2)
        SetTypicalTransferOperators(Core2)

        Core2(...){
            const auto result = init();
            if(!result.has_value()){
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
            
        }

        void run() noexcept{
            mainLoop();
        }

        ~Core2(){
            processDelete();
        }

    private:

        void mainLoop(){
            float lastSecond = 0.0f;

            recordFunction = [this](VkCommandBuffer& cmd , const uint32_t id)-> ErrorDataOutput<void>{

                recordCmd(cmd , id);

                return {};
            };

            while(!glfwWindowShouldClose(context->window->window)){
                glfwPollEvents();
                const float time  = glfwGetTime();
                dt = time - lastSecond;
                lastSecond = time;

                if(!context->user->openedMenu) context->user->assetinput(context->window->window , dt);
                const auto result = context->processQueueLoop(recordFunction);
                if(!result.has_value()){
                    engineLogger(result);
                    return;
                }
                if(firstIteration){
                    firstIteration = false;
                }
            }
        }

        void processCopy(Core2&& input) noexcept {

        }

        void processDelete() noexcept{
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
            pt = nullptr;
            rt = nullptr;
            ao = nullptr;
            forward = nullptr;
            render = nullptr;
            atmosphere = nullptr;
            meshes.clear();
            models.clear();
            buffers = nullptr;
            renderImages = nullptr;
            rc = nullptr;
            RCManager = nullptr;
            light_m = nullptr;
            tlas = nullptr;
            Mesh2::performDelete();
            manager = nullptr;
            context = nullptr;
            
        }

        [[nodiscard]] ErrorDataOutput<void> init() noexcept{
            {
                const auto result = createFolders();
                if(!result.has_value()) return result;
            }

            {
                const auto result = createAllSources();
                if(!result.has_value()) return result;
            }

            {
                const auto result = createMeshes();
                if(!result.has_value()) return result;
            }
            setDefaultSettings();

            {
                const auto result = createAllPasses();
                if(!result.has_value()) return result;
            }


            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> createFolders() noexcept{
            const std::string pipeCache = "../PipelineCache";
            const std::string shaderCache = "../ShaderModuleCache";
            try{

                namespace fs = std::filesystem;
                if(fs::exists(pipeCache) && fs::exists(shaderCache)) return {};

                fs::create_directory(pipeCache);
                fs::create_directory(shaderCache);
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create or check existance of folders for cache"});
            }

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> createAllPasses() noexcept{
            try{
                graph = std::make_shared<GraphRender2>();

                {
                    const auto result = graph->processInputParameter(light_m->lightBuffer.get() , "lightBuffer");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = graph->processInputParameter(renderImages.get());
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = graph->processInputParameter(buffers.get());
                    if(!result.has_value()) return result;
                }

                // now we create all the passes

                {
                    AtmosphereScattering::CreateInfo info{};
                    info.context = context.get();
                    info.skyView = getShaderPath("../Shaders/atmosphereScattering/SkyViewLUT");
                    info.transition = getShaderPath("../Shaders/atmosphereScattering/transitionCube");
                    atmosphere = std::make_shared<AtmosphereScattering>(info);
                }

                {
                    DeferredRender::CreateInfo info{};
                    info.context = context.get();
                    info.fragment = getShaderPath("../Shaders/DeferredRender/fragmentShader");
                    info.vertex = getShaderPath("../Shaders/DeferredRender/vertexShader");
                    info.manager = manager.get();

                    render = std::make_shared<DeferredRender>(info);
                }

                {
                    ForwardPlus::CreateInfo info{};
                    info.context = context.get();
                    info.path = getShaderPath("../Shaders/ForwardPlus/forward");
                    info.generalData = forwardData;

                    forward = std::make_shared<ForwardPlus>(info);
                }

                {
                    AmbientOcclusion::CreateInfo info{};
                    info.context = context.get();
                    info.path = getShaderPath("../Shaders/AmbientOcclusion/AmbientOcclusion");

                    ao = std::make_shared<AmbientOcclusion>(info);
                }

                {
                    RayQueryShadows::CreateInfo info{};
                    info.context = context.get();
                    info.as = tlas.get();
                    info.path = getShaderPath("../Shaders/RayQueryShadows/rayQueryShadows");
                    info.manager = manager.get();

                    shadows = std::make_shared<RayQueryShadows>(info);
                }

                {
                    BilateralFiltering::CreateInfo info{};
                    info.context = context.get();
                    info.path = getShaderPath("../Shaders/BF_singular/bilateralFilteringSingular");

                    bilateral = std::make_shared<BilateralFiltering>(info);
                }

                {
                    VolumentricRender::CreateInfo info{};
                    info.context = context.get();
                    info.path = getShaderPath("../Shaders/volumentricRender/fogCalc");

                    fog = std::make_shared<VolumentricRender>(info);
                }

                {
                    LightPass::CreateInfo info{};
                    info.context = context.get();
                    info.path = getShaderPath("../Shaders/LightPass/lightPass");

                    lightPass = std::make_shared<LightPass>(info);
                }

                {
                    IndirectLightSum::CreateInfo info{};
                    info.context = context.get();
                    info.path = getShaderPath("../Shaders/indirectShader/indirectCalc");

                    indirect = std::make_shared<IndirectLightSum>(info);
                }

                {
                    ReBLUR::CreateInfo info{};
                    info.context = context.get();
                    info.spatial = getShaderPath("../Shaders/denoiser/ReBLUR_spatial");
                    info.temporal = getShaderPath("../Shaders/denoiser/ReBLUR_temporal");

                    reblur = std::make_shared<ReBLUR>(info);
                }

                {
                    ReLAX::CreateInfo info{};
                    info.context = context.get();
                    info.spatial = getShaderPath("../Shaders/denoiser/ReLAX_spatial");
                    info.temporal = getShaderPath("../Shaders/denoiser/ReLAX_temporal");

                    relax = std::make_shared<ReLAX>(info);
                }

                {
                    PostProcess::CreateInfo info{};
                    info.context = context.get();
                    info.path = getShaderPath("../Shaders/PostProcess/PostProcess");

                    post = std::make_shared<PostProcess>(info);
                }

                {
                    AEHDR::CreateInfo info{};
                    info.context = context.get();
                    info.path = getShaderPath("../Shaders/HDR/autoExposure");

                    aehdr = std::make_shared<AEHDR>(info);
                }

                {
                    HDR::CreateInfo info{};
                    info.context = context.get();
                    info.path =  getShaderPath("../Shaders/HDR/HDR_mapped");
                    hdr = std::make_shared<HDR>(info);
                }

                {
                    imgui = std::make_shared<Imgui2>(context.get());
                }

                {
                    RadianceCascade::CreateInfo info{};
                    info.context = context.get();
                    info.as = tlas.get();
                    info.manager = manager.get();
                    info.probes = RCManager.get();
                    info.lights = light_m.get();
                    info.placementPath = getShaderPath("../Shaders/RadianceCascade/RCplacement");
                    info.rayGenerationPath = getShaderPath("../Shaders/RadianceCascade/RCgeneration");
                    info.missPath = getShaderPath("../Shaders/RadianceCascade/RCmiss");
                    info.shadowMissPath = getShaderPath("../Shaders/RadianceCascade/RCshadowMiss");
                    info.hitPath = getShaderPath("../Shaders/RadianceCascade/RChit");
                    info.shadowAnyHitPath = getShaderPath("../Shaders/RadianceCascade/RCshadowAnyHit");
                    info.shadowHitPath = getShaderPath("../Shaders/RadianceCascade/RCshadowHit");
                    info.alphaAnyHitPath = getShaderPath("../Shaders/RadianceCascade/RCalphaAnyHit");
                    info.combinePath = getShaderPath("../Shaders/RadianceCascade/CombineCascades");

                    rc = std::make_shared<RadianceCascade>(info);
                }

                {
                    RayTracing::CreateInfo info{};
                    info.as = tlas.get();
                    info.context = context.get();
                    info.generalHitPath = getShaderPath("../Shaders/RayTracing/generalHit");
                    info.generalMissPath = getShaderPath("../Shaders/RayTracing/generalMiss");
                    info.rayGenerationPath = getShaderPath("../Shaders/RayTracing/rayGeneration");
                    info.shadowMissPath = getShaderPath("../Shaders/RayTracing/shadowMiss");
                    info.shadowAnyHitPath = getShaderPath("../Shaders/RayTracing/shadowAnyHit");
                    info.shadowHitPath = getShaderPath("../Shaders/RayTracing/shadowHit");
                    info.giHitPath = getShaderPath("../Shaders/RayTracing/GI_hit");
                    info.giMissPath = getShaderPath("../Shaders/RayTracing/GI_miss");
                    info.generalAlphaAnyHitPath = getShaderPath("../Shaders/RayTracing/alphaAnyHit");
                    info.giAlphaAnyHitPath = getShaderPath("../Shaders/RayTracing/GI_alphaAnyHit");
                    info.manager = manager.get();
                    info.lights = light_m.get();
                    info.probes = RCManager.get();

                    rt = std::make_shared<RayTracing>(info);
                }

                {
                    // the path tracer : the primary rays start in its ray generation shader , so it replaces the deferred render and the ray tracing pass
                    PathTracing::CreateInfo info{};
                    info.as = tlas.get();
                    info.context = context.get();
                    info.manager = manager.get();
                    info.lights = light_m.get();
                    info.rayGenerationPath = getShaderPath("../Shaders/PathTracing/PTgen");
                    info.primaryMissPath = getShaderPath("../Shaders/PathTracing/PTmiss");
                    info.secondaryMissPath = getShaderPath("../Shaders/PathTracing/PTSecondaryMiss");
                    info.shadowMissPath = getShaderPath("../Shaders/PathTracing/shadowMiss");
                    info.primaryHitPath = getShaderPath("../Shaders/PathTracing/PThit");
                    info.secondaryHitPath = getShaderPath("../Shaders/PathTracing/PTSecondaryHit");
                    info.shadowAnyHitPath = getShaderPath("../Shaders/PathTracing/shadowAnyHit");
                    info.shadowHitPath = getShaderPath("../Shaders/PathTracing/shadowHit");
                    info.primaryAlphaAnyHitPath = getShaderPath("../Shaders/PathTracing/PTalphaAnyHit");
                    info.secondaryAlphaAnyHitPath = getShaderPath("../Shaders/PathTracing/PTSecondaryAlphaAnyHit");

                    pt = std::make_shared<PathTracing>(info);
                }

                const auto add = [this](GraphPass2* pass) -> ErrorDataOutput<void>{
                    return graph->addGraphPass2(pass , RasterizationPath);
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
                    imgui.get()
                };

                GraphPass2* orderRT[] = {
                    atmosphere.get() ,   // the sky : the sky view LUT and the cube
                    render.get() ,       // the g buffer
                    forward.get() ,      // the lamps of every tile
                    rc.get() ,           // Radiance Cascade : the probes on the surfaces , their rays , the merge of the cascades
                    rt.get(),
                    ao.get() ,           // the occlusion
                    bilateral.get() ,    // the filter of the shadows and of the occlusion
                    fog.get() ,          // the froxel volume of the fog
                    reblur.get() ,       // diffuse
                    relax.get() ,        // specular
                    post.get() ,         // diffuse * albedo + specular , the fog
                    aehdr.get() ,        // the exposure
                    hdr.get() ,          // the tone mapping into the final image
                    imgui.get()
                };

                // the same passes as the ray tracing , but without Radiance Cascade ( the path tracer has its own bounces )
                GraphPass2* orderPT[] = {
                    atmosphere.get() ,   // the sky : the sky view LUT and the cube
                    render.get() ,       // the g buffer ( the raster has the mip levels of the textures : the primary hit of the path tracer reads it )
                    forward.get() ,      // the lamps of every tile
                    pt.get() ,           // the path tracer : the light and the bounces of the pixels of the g buffer , the sky
                    ao.get() ,           // the occlusion
                    bilateral.get() ,    // the filter of the shadows and of the occlusion
                    fog.get() ,          // the froxel volume of the fog
                    reblur.get() ,       // diffuse
                    relax.get() ,        // specular
                    post.get() ,         // diffuse * albedo + specular , the fog
                    aehdr.get() ,        // the exposure
                    hdr.get() ,          // the tone mapping into the final image
                    imgui.get()
                };

                for(GraphPass2* pass : order){
                    const auto result = add(pass);
                    if(!result.has_value()) return result;
                }

                for(GraphPass2* pass : orderRT){
                    const auto result = graph->addGraphPass2(pass , RayTracingPath);
                    if(!result.has_value()) return result;
                }

                for(GraphPass2* pass : orderPT){
                    const auto result = graph->addGraphPass2(pass , PathTracingPath);
                    if(!result.has_value()) return result;
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create all sources"});
            }
            return {};
        }

        void setDefaultSettings() noexcept{
            generalSettings = GeneralInfo{};
            generalSettings.exposure = 1.6f;
            generalSettings.directWeight = 0.5f;
            generalSettings.indirectWeight = 0.5f;
            generalSettings.ambientInIndirect = 1;
            generalSettings.normalMapOpenGL = 1;
            generalSettings.ReBLUR = 1;
            generalSettings.ReLAX = 1;

            // features
            featureSettings = GeneralFeatures{};
            featureSettings.AO = SSAO_REQUIRED;
            featureSettings.AO_iteration = 16;
            featureSettings.AOradius = 0.6f;
            featureSettings.AO_intens = 1.0f;
            featureSettings.indirectMode = SSGI_REQUIRED;
            featureSettings.VSM = 0;
            featureSettings.RayQuery = 1;
            featureSettings.RayQueryLightSampleCount = 4;
            featureSettings.RayQuerySunSampleCount = 4;
            featureSettings.shadowThickness = 1.0f;
            featureSettings.temporalShadowK = 0.85f;
            featureSettings.light_sun_Proportion = 0.5f;
            featureSettings.RT = 1;
            featureSettings.PT = 0;
            featureSettings.currentMode = DEFAULT_MODE;
            featureSettings.randomValues = 1;
            featureSettings.volumentricFog = 0;
            featureSettings.ReSTIR = 1;
            featureSettings.IndirectShare = 0.5f;
            featureSettings.maxDepth = 50.0f;
            featureSettings.ReSTIRGI = true;
            featureSettings.ReSTIRGI_temporal = true;
            featureSettings.ReSTIRGI_spatial = true;
            featureSettings.lightShadowFade = 0.5f;
            featureSettings.skyboxSampleCount = 2u;
            featureSettings.sunShadowFade = 30.0f;
            featureSettings.shadowMinimum = 0.2f;

            fogSettings = FogVolume{};
            fogSettings.center = glm::vec3(0.0f);
            fogSettings.halfExtent = glm::vec3(1000.0f);
            fogSettings.albedo = glm::vec3(1.0f);
            fogSettings.density = 0.012f;
            fogSettings.feather = 0.0f;
            fogSettings.g = 0.45f;
            fogSettings.gridSize = fogGridSize;
            fogSettings.heightFallOff = 0.12f;
            fogSettings.fogBaseHeight = FLOOR_Y;
            fogSettings.color = fogTint * fogGlow;

            // the sun : 40 degrees above the horizon , the azimuth 215 degrees . the light travels away from the sun
            {
                const float elevation = glm::radians(40.0f);
                const float azimuth = glm::radians(215.0f);
                const glm::vec3 toSun(std::cos(elevation) * std::cos(azimuth) , std::sin(elevation) , std::cos(elevation) * std::sin(azimuth));

                sunSettings = SunSource{};
                // the size of the sun for the shadow ( the half angle of the cone , radians ) : the real sun is 0.00465 , a bigger one makes a softer penumbra
                sunSettings.sourceRadius = 0.00465f;
                sunSettings.direction = -toSun;
                sunSettings.intensity = 4.0f;
                sunSettings.diffuse = glm::vec3(1.0f , 0.95f , 0.85f) * sunSettings.intensity;

                const float horizontal = std::sqrt(toSun.x * toSun.x + toSun.z * toSun.z);
                skySettings = SkyViewData{};
                skySettings.cameraHeight = 5.0f;
                skySettings.sunCosZenith = toSun.y;
                skySettings.sunCosAzimuth = toSun.x / horizontal;
                skySettings.sunSinAzimuth = toSun.z / horizontal;
            }

            // Forward+ : the tiles of the screen and the number of the lights . the pass takes forwardData when it is created , and updateData sends it to the buffer
            forwardData = ForwardPlusInfo{};
            forwardData.TILE_SIZE = 16;
            forwardData.TILE_PER_ROW = (context->sc->extent.width + forwardData.TILE_SIZE - 1) / forwardData.TILE_SIZE;
            forwardData.TILE_TOTAL_AMOUNT = forwardData.TILE_PER_ROW * ((context->sc->extent.height + forwardData.TILE_SIZE - 1) / forwardData.TILE_SIZE);
            forwardData.totalLights = lightCount;

            requiredMode = RayTracingRequired;
        }

        [[nodiscard]] ErrorDataOutput<void> createAllSources() noexcept{

            try{
                struct VulkanContext::CreateInfo info{};
                info.asyncMultiThreadQueueCount =4;
                info.multithreadQueueCount = 1;
                context = std::make_shared<VulkanContext>(info);
                renderImages = std::make_shared<FrameTargets>(context.get());
                buffers = std::make_shared<SourceBuffers>(context.get());
                {
                    TextureManager2::CreateInfo info{};
                    info.context = context.get();
                    manager = std::make_shared<TextureManager2>(info);
                }

                {
                    LightManager::CreateInfo info{};
                    info.context = context.get();
                    info.shaderPath = getShaderPath("../Shaders/LightGridUpdate/LightGridUpdate");
                    info.user = context->user.get();

                    light_m = std::make_shared<LightManager>(info);
                }

                {
                    RCProbeManager::CreateInfo info{};
                    info.context = context.get();

                    RCManager = std::make_shared<RCProbeManager>(info);
                }

                // multiscatter
                MultiScatter::CreateInfo mlInfo{};
                mlInfo.context = context.get();
                mlInfo.image = renderImages->multiScatter.get();
                mlInfo.path = getShaderPath("../Shaders/MultiScatter/MultiScatter");

                MultiScatter scatter(mlInfo);

                // transmission LUT
                TransmissionLUTPipe::CreateInfo transInfo{};
                transInfo.context = context.get();
                transInfo.image = renderImages->transmissionLUT.get();
                transInfo.path = getShaderPath("../Shaders/atmosphereScattering/transmissionLUT");

                TransmissionLUTPipe trans(transInfo);

                // multiscatterLUT
                MultiScatterLUTPipe::CreateInfo mulLUTInfo{};
                mulLUTInfo.context = context.get();
                mulLUTInfo.path = getShaderPath("../Shaders/atmosphereScattering/multiScattering");
                mulLUTInfo.transmission = renderImages->transmissionLUT.get();
                mulLUTInfo.image = renderImages->multiScatterLUT.get();

                MultiScatterLUTPipe multi(mulLUTInfo);
                vkDeviceWaitIdle(context->logicalDevice->device);
            
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create all sources"});
            }

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> createMeshes() noexcept{
            try{

                {
                    const auto result = Mesh2::Begin(context.get() , manager.get());
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = createLights();
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = createModels();
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = Mesh2::End();
                    if(!result.has_value()) return result;
                }

                tlas = std::make_shared<TLAS2>(context.get());
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create all sources"});
            }

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> createModels() noexcept{
            try{

                // stone 1
                {
                    // the fort from glTF : the node transforms are baked into the vertices , the textures are found next to the .gltf
                    Model3::CreateInfo info{};
                    info.color = glm::vec3(0.0f);
                    info.path = "../models/Sponza/Sponza.gltf";
                    info.metallic = 0.0f;
                    info.roughness = 1.0f;
                    models.push_back(std::make_shared<Model3>(info));
                    glm::mat4 model(1.0f);
                    model = glm::translate(model , glm::vec3(0.0f ,10.0f, 0.0f));
                    model = glm::scale(model , glm::vec3(2.0f));
                    models[0]->setMatrix(model , 0);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create models"});
            }
            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> createLights() noexcept{
            // 3 lights
            {
                std::vector<lg::LightSource> lights(3);

                std::mt19937 rng{std::random_device{}()};
                const auto rnd = [&rng](const float low , const float high){
                    // the order of the limits does not matter ( the distribution asserts when low is bigger than high , and the program is aborted )
                    return std::uniform_real_distribution<float>(std::min(low , high) , std::max(low , high))(rng);
                };

                // the first two : violet , not too bright , a random strength and reach
                for(uint32_t i = 0 ; i < 3 ; i++){
                    const float intensity = rnd(4.0f , 10.0f);
                    lights[i].diffuse = glm::vec3(rnd(0.1f , 1.0f) , rnd(0.0f , 1.0f) , rnd(0.1f , 1.0f)) * intensity;
                    lights[i].radius = rnd(15.0f , 30.0f);
                    lights[i].Linear = rnd(0.05f , 0.012f);
                    lights[i].Quadratic = rnd(0.02f , 0.004f);
                    // the size of the lamp itself ( meters ) : the penumbra of its shadow
                    lights[i].sourceRadius = rnd(1.0f , 5.0f);
                }



                lights[0].position = glm::vec3(-3.7f , 13.0f , -8.7f);
                lights[1].position = glm::vec3(-13.5f , 20.4f , -7.2f);
                lights[2].position = glm::vec3(15.5f , 12.6f , 7.2);
                // the names : the lamps are found by them , not by the index in a vector . the grid and the buffers are written by light_m->updateGrid every frame
                for(uint32_t i = 0 ; i < lights.size() ; i++){
                    light_m->addLight(lights[i] , std::format("light_{}" , i));
                }
            }
            return {};
        }

        void updateLightPoses(const float dt) noexcept
        {
            
        }

        void recordCmd(VkCommandBuffer& cmd , const uint32_t index) noexcept{
            {
                VkCommandBufferBeginInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                info.pInheritanceInfo = nullptr;
                info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                info.pNext = nullptr;

                const auto result = vkBeginCommandBuffer(cmd , &info);
            }
            updateData(cmd);

            renderImages->cleanImages(cmd);

            if(requiredMode == RasterizationRequired){
                const auto result = graph->execute(cmd , RasterizationPath);
                if(!result.has_value()){
                    vkEndCommandBuffer(cmd);
                    return;
                }
            }else if(requiredMode == RayTracingRequired){
                const auto result = graph->execute(cmd , RayTracingPath);
                if(!result.has_value()){
                    vkEndCommandBuffer(cmd);
                    return;
                }
            }else if(requiredMode == PathTracingRequired){
                const auto result = graph->execute(cmd , PathTracingPath);
                if(!result.has_value()){
                    vkEndCommandBuffer(cmd);
                    return;
                }
            }

            VkImageMemoryBarrier2 swapchainBarrier{};
            swapchainBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            swapchainBarrier.pNext = nullptr;
            swapchainBarrier.image = context->images[index];
            swapchainBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT , 0 , 1 , 0 , 1};

            // the final image is read by the copy , the swapchain image is written by it
            {
                std::array<VkImageMemoryBarrier2, 2> barriers;
                barriers[0] = renderImages->finalImage->translateImage(VK_ACCESS_2_TRANSFER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

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

                vkCmdCopyImage(cmd , renderImages->finalImage->image , VK_IMAGE_LAYOUT_GENERAL, context->images[index] , VK_IMAGE_LAYOUT_GENERAL , 1 , &region);
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
        }



        void updateData(VkCommandBuffer& cmd) noexcept {
            // not made again every frame : the UI owns exposure , weights , features , fog , sun
            GeneralInfo& data = generalSettings;
            GeneralFeatures& features = featureSettings;
            FogVolume& fogData = fogSettings;
            SkyViewData& skyData = skySettings;
            SunSource& sun = sunSettings;
            const float aspect = static_cast<float>(context->sc->extent.width) / static_cast<float>(context->sc->extent.height);

            glm::mat4 proj = glm::perspective(glm::radians(70.0f) , aspect , 0.001f , 200.0f);
            glm::mat4 view = glm::lookAt(context->user->position , context->user->position + context->user->direction , glm::vec3(0.0f , 1.0f , 0.0f));
            features.currentMode = context->user->currentMode;
            data.inverseView = glm::inverse(view);
            data.inverseProj = glm::inverse(proj);
            data.viewProj = proj * view;
            data.playerDirection = context->user->direction;
            data.playerPosition = context->user->position;
            data.playerNear = 0.001f;
            data.playerFar = 200.0f;
            data.dt = dt;
            data.frameIndex = context->sc->frameIndex;
            data.screenWidth = context->sc->extent.width;
            data.screenHeight = context->sc->extent.height;
            data.viewOnly = view;
            data.projOnly = proj;
            data.invViewProj = glm::inverse(data.viewProj);
            data.lightCount = lightCount;
            if(firstIteration){
                data.prevViewProj = data.viewProj;
            }else{
                data.prevViewProj = prevVP;
            }

            prevVP = data.viewProj;

            // the height of the camera over the ground is the only field of the sky data that the application owns , it is made every frame ( the sun header can be closed )
            skyData.cameraHeight = std::max(context->user->position.y - FLOOR_Y , 0.5f);

            drawImGui(data , features , fogData , skyData , sun);

            // the Forward+ has no buffer of the features : it takes the fade of the shadows from its own info ( after the ImGui , it may change the slider )
            forwardData.lightShadowFade = features.lightShadowFade;

            buffers->generalBuffer->updateBufferCmd(cmd ,data , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            buffers->sunBuffer->updateBufferCmd(cmd ,sun , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            buffers->fogData->updateBufferCmd(cmd ,fogData , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            buffers->skyBuffer->updateBufferCmd(cmd ,skyData , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            buffers->features->updateBufferCmd(cmd ,features , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            buffers->forwardInfo->updateBufferCmd(cmd ,forwardData , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );

            light_m->updateGrid(cmd);

            Mesh2::updateMeshData(cmd);
            tlas->updateTLAS(cmd);
        }

        // here will be the input data
        void drawImGui(GeneralInfo &general , GeneralFeatures &features , FogVolume& fogData , SkyViewData &skyData , SunSource &sunData) noexcept {
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            
            gui::NewFrame();
            gui::Begin("generalSettings");
            gui::Text("FPS :  %.1f , ms : %.3f " , general.dt > 1e-6f ? 1.0f / general.dt : 0.0f , general.dt * 1000.0f);
            gui::Text("Player Position : X > %.1f  : Y > %.1f : Z >  %.1f" , context->user->position.x, context->user->position.y ,context->user->position.z);

            gui::Separator();

            if(gui::CollapsingHeader("General data")){
                drawSettingsForGeneralData(general);
            }

            gui::Separator();

            if(gui::CollapsingHeader("Fog settings")){
                drawSettingsForFog(fogData);
            }

            gui::Separator();

            if(gui::CollapsingHeader("Sun settings")){
                drawSettingsForSun(sunData,  skyData);
            }
            gui::Separator();

            if(gui::CollapsingHeader("Features")){
                drawSettingsForFeatures(features);
            }

            gui::Separator();

            gui::End();
            gui::Render();
        }

        void drawSettingsForGeneralData(GeneralInfo &info) noexcept{
            // exposure
            // weights for direct and indirect
            gui::Text("this settings are the basic, there is no really a big diff here");
            gui::Text("A general exposure , nothing really new");
            gui::SliderFloat("Exposure correction" , &info.exposure , 0.0f, 10.0f);
            gui::Separator();
            // the proportion is the weight of the indirect light : it is kept in the data itself , so the slider does not go back to zero every frame
            float prop = info.indirectWeight;
            gui::Text("This propotion defines the proportion between direct and indirect");
            if(gui::SliderFloat("Light Proportion" , &prop , 0.0f , 1.0f)){
                info.directWeight = 1.0f - prop;
                info.indirectWeight = prop;
            }
            gui::Separator();
            // the green channel of the normal maps : OpenGL ( up ) or DirectX ( down ) . the raster and the RT shaders both read it
            bool normalMapOpenGL = info.normalMapOpenGL != 0u;
            if(gui::Checkbox("Normal maps are OpenGL style" , &normalMapOpenGL)){
                info.normalMapOpenGL = normalMapOpenGL ? 1u : 0u;
            }

            bool blur = info.ReBLUR;
            bool lax = info.ReLAX;
            if(gui::Checkbox("Use ReBLUR" , &blur)){
                info.ReBLUR = blur ? 1u : 0u;
            }

            if(gui::Checkbox("Use ReLAX" , &lax)){
                info.ReLAX = lax ? 1u : 0u;
            }
        }

        void drawSettingsForFog(FogVolume &info) noexcept{
            gui::Text("this settings are for fog");
            
            gui::Text("Density : i think this doesnt require explanation");
            // a real fog is thin : 0.001 is a haze , 0.05 is a thick fog . the slider is logarithmic ( the step grows with the value ) and the range is small ,
            // so the small values are not squeezed into the first pixels of the slider . Ctrl + click types the number
            gui::SliderFloat("Density" , &info.density , 0.0f, 0.2f , "%.4f" , ImGuiSliderFlags_Logarithmic);

            gui::Separator();
            gui::Text("FallOff : how fast fog disapears per metr");
            gui::SliderFloat("HeightFallOff" , &info.heightFallOff , 0.0f, 1.0f , "%.3f" , ImGuiSliderFlags_Logarithmic);

            gui::Separator();
            gui::Text("baseHeight : at which point the fog is the most dens");
            // a drag : 0.05 m per pixel of the mouse , slow and precise
            gui::DragFloat("baseHeight" , &info.fogBaseHeight , 0.05f , -40.0f , 40.0f , "%.2f m");

            {
                gui::Separator();
                gui::Text("albedo : how does the light behaves in fog");
                // the colour is edited in place : the three floats of the vector
                gui::ColorEdit3("Albedo " , &info.albedo.x);
            }
            
            {
                // the glow of the fog is the product of the colour and the intensity : both are kept ( fogTint , fogGlow ) , so the colour is not lost when the glow is 0
                gui::Separator();
                gui::Text("fog Color : color of fog");
                bool changed = gui::ColorEdit3("Fog Color " , &fogTint.x);

                gui::Separator();
                gui::Text("fog Glow : same here , no explanation");
                changed |= gui::SliderFloat("Fog Glow" , &fogGlow , 0.0f , 2.0f , "%.3f" , ImGuiSliderFlags_Logarithmic);

                if(changed) info.color = fogTint * fogGlow;
            }

            gui::Separator();
            gui::Text("G : a twhich direction does the light go");
            gui::SliderFloat("G" , &info.g , -0.95f , 0.95f , "%.2f");
        }
    
        void drawSettingsForSun(SunSource &sun , SkyViewData& sky) noexcept{

            float zenith = glm::degrees(std::acos(std::clamp(sky.sunCosZenith , -1.0f , 1.0f)));   // 0 .. 180 : 0 - overhead , 90 - horizon , more - under it
            float azimuth = glm::degrees(std::atan2(sky.sunSinAzimuth , sky.sunCosAzimuth));       // -180 .. 180
            if(azimuth < 0.0f) azimuth += 360.0f;                                                   // 0 .. 360 , 0 is +X

            static glm::vec3 diffuse = {1.0f , 1.0f , 1.0f};
            static float intensity = glm::sin(zenith);
            float color[3] = {diffuse.x , diffuse.y, diffuse.z};
            if(intensity > 1e-6f){
                color[0] = diffuse.x;
                color[1] = diffuse.y;
                color[2] = diffuse.z;
            }

            bool changed = false;

            gui::Text("Zenith : the vertical orientation of the sun");
            changed |= gui::SliderFloat("Zenith" , &zenith , 0.0f , 97.5f , "%.1f deg");
            gui::Separator();

            gui::Text("Azimuth : the horizontal orientation of the sun");
            changed |= gui::SliderFloat("Azimuth" , &azimuth , 0.0f , 360.0f , "%.1f deg");
            gui::Separator();

            gui::Text("Sun color : the color of the sun");
            changed |= gui::ColorEdit3("Sun color" , color);
            gui::Separator();
            diffuse = {color[0] , color[1] , color[2]};
            // the size of the sun for the shadow : the half angle of the cone , radians ( the real sun is 0.00465 ) . the bigger , the softer the penumbra
            gui::SliderFloat("Sun source radius" , &sun.sourceRadius , 0.0f , 0.1f , "%.4f");

            if(changed){
                const float theta = glm::radians(zenith);
                const float phi = glm::radians(azimuth);

                const glm::vec3 toSun(std::sin(theta) * std::cos(phi) , std::cos(theta) , std::sin(theta) * std::sin(phi));
                intensity = glm::cos(glm::radians(glm::clamp(zenith , 0.0f  ,90.0f))) * 2.0f;
                sun.direction = -toSun;                 // the light travels away from the sun
                sun.intensity = intensity;
                sun.diffuse = glm::vec3(color[0] , color[1] , color[2]) * intensity;

                sky.sunCosZenith = std::cos(theta);     // the same as toSun.y
                sky.sunCosAzimuth = std::cos(phi);
                sky.sunSinAzimuth = std::sin(phi);
            }

            sky.cameraHeight = std::max(context->user->position.y - FLOOR_Y , 0.5f);
        }

        void drawSettingsForFeatures(GeneralFeatures &info) noexcept{

            // AO
            gui::BeginDisabled(false);
            {
                // type
                if(gui::Button("SSAO")){
                    info.AO = SSAO_REQUIRED;
                }
                gui::SameLine();
                if(gui::Button("HBAO")){
                    info.AO = HBAO_REQUIRED;
                }
                gui::SameLine();
                if(gui::Button("NO AO")){
                    info.AO = AO_NOT_REQUIRED;
                }
                gui::BeginDisabled(info.AO == AO_NOT_REQUIRED);
                if(gui::CollapsingHeader("ao parameters"))
                {
                    gui::Separator();
                    ImGui::Spacing();
                    // intes
                    gui::SliderFloat("AO intesivity" , &info.AO_intens , 0.0f , 5.0f);
                    gui::Separator();
                    ImGui::Spacing();

                    // rate
                    gui::Separator();
                    gui::SliderInt("AO SampleCount" , std::bit_cast<int*>(&info.AO_iteration), 0 , 64);
                    gui::Separator();
                    ImGui::Spacing();

                    gui::Separator();
                    gui::SliderFloat("AO Radius" , &info.AOradius, 0 , 20.0f);
                    gui::Separator();
                }
                gui::EndDisabled();
            }

            gui::EndDisabled();
            gui::Separator();

            // shadows
            if(ImGui::CollapsingHeader("shader Parameters"))
            {
                {
                    gui::SliderInt("Sun SampleCount" , std::bit_cast<int*>(&info.RayQuerySunSampleCount), 0 , 64);
                    gui::Separator();
                    ImGui::Spacing();

                    gui::SliderInt("LightPos SampleCount" , std::bit_cast<int*>(&info.RayQueryLightSampleCount), 0 , 64);
                    gui::Separator();
                    ImGui::Spacing();

                    gui::SliderFloat("light to Sun proportion" , &info.light_sun_Proportion , 0.0f , 1.0f);
                    gui::Separator();
                    ImGui::Spacing();

                    gui::SliderFloat("shadow thickness" , &info.shadowThickness , 0.0f , 5.0f);
                    gui::Separator();
                    ImGui::Spacing();

                    gui::SliderFloat("temporal occlusion" , &info.temporalShadowK , 0.0f , 1.0f);
                    gui::Separator();
                    ImGui::Spacing();

                    // how far beyond the radius of a lamp its shadow still counts ( in radii ) : 0 - it ends at the radius with a jump
                    gui::SliderFloat("light shadow fade" , &info.lightShadowFade , 0.0f , 2.0f);
                    gui::Separator();

                    // RT : the shadow of the sun gets weaker with the distance to what covers the pixel ( meters , 0 - no fade )
                    gui::SliderFloat("sun shadow fade" , &info.sunShadowFade , 0.0f , 200.0f);
                    gui::Separator();

                    // how dark a shadow of the direct light can get : 0 - black , 1 - no shadow
                    gui::SliderFloat("shadow minimum" , &info.shadowMinimum , 0.0f , 1.0f);
                    gui::Separator();
                }
            }
            gui::Separator();

            // RT PT rast

            {
                if(gui::Button("RASTERIZATION")){
                    info.RT = 0;
                    info.PT = 0;
                    requiredMode = RasterizationRequired;
                }
                gui::SameLine();
                
                if(gui::Button("RAY TRACING")){
                    info.RT = 1;
                    info.PT = 0;
                    requiredMode = RayTracingRequired;
                }
                gui::SameLine();

                if(gui::Button("PATH TRACING")){
                    info.RT = 0;
                    info.PT = 1;
                    requiredMode = PathTracingRequired;
                }
            }
            gui::Separator();
            ImGui::Spacing();

            // SSR / SSGI
            gui::BeginDisabled(info.PT && info.RT);
            {
                if(gui::Button("SSR")){
                    info.indirectMode = SSR_REQUIRED;
                }
                gui::SameLine();

                if(gui::Button("SSGI")){
                    info.indirectMode = SSGI_REQUIRED;
                }
                gui::SameLine();

                if(gui::Button("NO MODE")){
                    info.indirectMode = NON_REQUIRED;
                }
            }
            gui::EndDisabled();
            gui::Separator();

            gui::BeginDisabled(info.PT);
            {
                if(gui::Button(info.ReSTIR ? "ReSTIR DI : ON" : "ReSTIR DI : OFF")){
                    info.ReSTIR = !info.ReSTIR;
                }
            }
            gui::EndDisabled();
            gui::Separator();

            gui::BeginDisabled(!info.RT);
            if(gui::CollapsingHeader("GI"))
            {
                // ReSTIR DI : the best lamp of the tile , or the sum of all the lamps ( the exact one )
                {
                    if(gui::Button(info.ReSTIRGI ? "ReSTIR GI : ON" : "ReSTIR GI : OFF")){
                        info.ReSTIRGI = !info.ReSTIRGI;
                        info.RadianceCascade = false;
                    }
                }
                gui::Separator();

                gui::BeginDisabled(!info.ReSTIRGI);
                {
                    if(gui::Button(info.ReSTIRGI_temporal ? "ReSTIR GI temporal: ON" : "ReSTIR GI temporal: OFF")){
                        info.ReSTIRGI_temporal = !info.ReSTIRGI_temporal;
                    }
                }
                gui::EndDisabled();
                gui::Separator();

                gui::BeginDisabled(!info.ReSTIRGI);
                {
                    if(gui::Button(info.ReSTIRGI_spatial ? "ReSTIR GI spatial: ON" : "ReSTIR GI spatial: OFF")){
                        info.ReSTIRGI_spatial = !info.ReSTIRGI_spatial;
                    }
                }
                gui::EndDisabled();
                gui::Separator();

                {
                    if(gui::Button(info.RadianceCascade ? "Radiance Cascade: ON" : "Radiance Cascade: OFF")){
                        info.RadianceCascade = !info.RadianceCascade;
                        info.ReSTIRGI = false;
                    }
                }
            }
            gui::EndDisabled();
            gui::Separator();




            // fog render
            gui::BeginDisabled(false);
            {
                if(gui::Button("VOLUMENTRIC RENDER")){
                    info.volumentricFog = !info.volumentricFog;
                }
            }
            gui::EndDisabled();
        }


        pVulkanContext context; // done
        pFrameTargets renderImages; // done
        pSourceBuffers buffers; // done
        pTLAS2 tlas{}; // created
        pTextureManager2 manager; // created
        pLightManager light_m;
        pRCProbeManager RCManager;

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
        ForwardPlusInfo forwardData{};
        pGraphRender2 graph{};
        // here will be imgui
        pImgui2 imgui{};
        pRayTracing rt{};
        pPathTracing pt{};
        pRadianceCascade rc{};

        std::vector<pMesh2> meshes{};
        std::vector<pModel3> models{};

        glm::mat4 prevVP{1.0f};
        bool firstIteration{true};
        QueueRecordTypeFunction recordFunction{};
        const float fogGridSize = 64.0f;
        static constexpr float FLOOR_Y = -2.0f; 
        float dt = 0.0f;
        uint32_t requiredMode = RasterizationRequired;

        static constexpr uint32_t lightCount = 4;

        GeneralInfo generalSettings{};
        GeneralFeatures featureSettings{};
        FogVolume fogSettings{};
        SkyViewData skySettings{};
        SunSource sunSettings{};
        glm::vec3 fogTint{0.35f , 0.45f , 0.60f};
        float fogGlow{0.25f};
    };
}