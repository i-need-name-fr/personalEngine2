#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../Object2/Mesh2.hpp"

namespace mox{

    class HDR : public GraphPass2{
    public:
        HDR() = delete;
        HDR(HDR& input) = delete;
        HDR(const HDR& input) = delete;
        HDR& operator=(HDR& input) = delete;
        HDR& operator=(const HDR& input) = delete;

        HDR(HDR&& input) noexcept{
            processCopy(std::move(input));

        }

        HDR& operator=(HDR&& input) noexcept{
            processCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context{nullptr};
            std::string_view path;
        };

        explicit HDR(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeQueueManager | VkReqTypeLogicalDevice | VkReqTypePhysicalDevice , info.context);
            if(!result.has_value()){
                engineLogger(result.error());
                THROW_MESSAGE;
            }

            if(info.path.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create HDR , since the path to file is empty"});
                THROW_MESSAGE;
            }

            context = info.context;
            path = std::string(info.path);
        }

        ~HDR(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(16);

            reqs[0]= {moxGraphTypeImage , ImageType::RenderImage2 , "normalImage_0"};
            reqs[1]= {moxGraphTypeImage , ImageType::RenderImage2 , "normalImage_1"};

            reqs[2]= {moxGraphTypeImage , ImageType::RenderImage2 , "depthImage_0"};
            reqs[3]= {moxGraphTypeImage , ImageType::RenderImage2 , "depthImage_1"};

            reqs[4]= {moxGraphTypeImage , ImageType::RenderImage2 , "HDRImage_0"};
            reqs[5]= {moxGraphTypeImage , ImageType::RenderImage2 , "HDRImage_1"};

            reqs[6]= {moxGraphTypeImage , ImageType::RenderImage2 , "roughnessImage"};
            reqs[7]= {moxGraphTypeImage , ImageType::RenderImage2 , "metallicImage"};
            reqs[8]= {moxGraphTypeImage , ImageType::RenderImage2 , "albedoImage"};
            reqs[9]= {moxGraphTypeImage , ImageType::RenderImage2 , "pixelCheck"};
            reqs[10]= {moxGraphTypeImage , ImageType::RenderImage2 , "finalImage"};

            reqs[11].type = moxGraphTypeBuffer;
            reqs[11].name = "generalBuffer";

            reqs[12].type = moxGraphTypeBuffer;
            reqs[12].name = "generalFeatures";

            reqs[13].type = moxGraphTypeBuffer;
            reqs[13].name = "luminanceHistory";

            // the AO_MODE shows it
            reqs[14]= {moxGraphTypeImage , ImageType::RenderImage2 , "filteredAmbientOcclusion"};

            // the SHADOW_MODE shows it
            reqs[15]= {moxGraphTypeImage , ImageType::RenderImage2 , "filteredShadowImage"};

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]]  ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [HDR]"});

            const uint8_t currentImage = context->sc->frameIndex %2;

            const uint32_t x = (context->sc->extent.width + 7) / 8;
            const uint32_t y = (context->sc->extent.height + 7) / 8;

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);

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

            vkCmdDispatch(cmd , x , y, 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            return {};
        }

        [[nodiscard]] [[gnu::hot]]  const bool requestExecution() const noexcept override final{

            return true;
        }

        [[nodiscard]]  const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindAttributes(sources);
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]]  const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to translateImage ,since the pipeline wasnt created yet [HDR]"});
            const uint8_t currentImage = context->sc->frameIndex %2;

            std::array<VkBufferMemoryBarrier2 , 3> bufBar;
            std::array<VkImageMemoryBarrier2 , 10> imageBar;

            bufBar[0] = buffer("luminanceHistory")->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            bufBar[1] = buffer("generalBuffer")->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            bufBar[2] = buffer("generalFeatures")->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            imageBar[0] = image("albedoImage")->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[1] = image("normalImage" , currentImage)->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[2] = image("HDRImage" , currentImage)->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[3] = image("pixelCheck")->translateImage(VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[4] = image("depthImage" , currentImage)->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[5] = image("roughnessImage")->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[6] = image("metallicImage")->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[7] = image("finalImage")->translateImageUndefined(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[8] = image("filteredAmbientOcclusion")->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[9] = image("filteredShadowImage")->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 3;
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

        VulkanContext* context{nullptr};
        std::string path;
        pShaderPipeType pipe{nullptr};

        void processCopy(HDR&& input) noexcept{
            if(this == &input) return;
            processDelete();
            GraphPass2::operator=(std::move(input));

            context = std::exchange(input.context , nullptr);
            path = std::move(input.path);
            pipe = std::move(input.pipe);
            completedInit = std::exchange(input.completedInit , false);
        }

        void processDelete() noexcept{
            clearAttributes();
            context = nullptr;
            path.clear();
            pipe = nullptr;
            completedInit = false;
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            try{
                pipe = std::make_shared<ShaderPipeType>(context);

                {
                    std::vector<DescriptorLayoutData> data(13);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[5] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[6] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[7] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[8] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[9] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[10] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[11] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[12] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

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
                        std::vector<DescriptorData2> data(13);
                        data[0] = DescriptorData2::createData(buffer("luminanceHistory") , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0);
                        data[1] = DescriptorData2::createData(buffer("generalBuffer") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0);
                        data[2] = DescriptorData2::createData(buffer("generalFeatures") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0);

                        data[3] = DescriptorData2::createData(image("albedoImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);
                        data[4] = DescriptorData2::createData(image("normalImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 4 , 0);
                        data[5] = DescriptorData2::createData(image("HDRImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 5 , 0);
                        data[6] = DescriptorData2::createData(image("pixelCheck") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 6 , 0);
                        data[7] = DescriptorData2::createData(image("depthImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 7 , 0);
                        data[8] = DescriptorData2::createData(image("roughnessImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0);
                        data[9] = DescriptorData2::createData(image("metallicImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 9 , 0);
                        data[10] = DescriptorData2::createData(image("finalImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 10 , 0);
                        data[11] = DescriptorData2::createData(image("filteredAmbientOcclusion") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 11 , 0);
                        data[12] = DescriptorData2::createData(image("filteredShadowImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 12 , 0);

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
                    info.nextStage = 0;
                    info.filePath = path;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};
                    
                    pipe->addShader(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create HDR pipeline"});
            }
            completedInit = true;
            return {};
        }
    };

    using pHDR = std::shared_ptr<HDR>;
}