#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Sampler/Sampler.hpp"

namespace mox{

    // the last step before the tone mapping : diffuse * albedo + specular ( both are already denoised ) , then the fog , into HDRImage
    // shader : PostProcess/PostProcess.hlsl
    // the sky pixels ( pixelCheck ) and the path tracer take diffuseImage as it is
    class PostProcess : public GraphPass2{
    public:

        SetTypicalOperators(PostProcess)
        SetTypicalTransferOperators(PostProcess)

        struct CreateInfo{
            VulkanContext* context;
            std::string path;
        };

        explicit PostProcess(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            if(info.path.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create PostProcess , since the path to the shader is empty"});
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~PostProcess(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs{};
            reqs.reserve(20);

            const auto image = [&reqs](const char* name , const ImageType type){
                GraphParameterInfo info{};
                info.type = moxGraphTypeImage;
                info.name = name;
                info.imageSpecification.type = type;
                reqs.push_back(info);
            };

            const auto buffer = [&reqs](const char* name){
                GraphParameterInfo info{};
                info.type = moxGraphTypeBuffer;
                info.name = name;
                reqs.push_back(info);
            };

            // the ping-pong pairs
            image("diffuseImage_0" , ImageType::RenderImage2);
            image("diffuseImage_1" , ImageType::RenderImage2);
            image("specularImage_0" , ImageType::RenderImage2);
            image("specularImage_1" , ImageType::RenderImage2);
            image("depthImage_0" , ImageType::RenderImage2);
            image("depthImage_1" , ImageType::RenderImage2);
            image("indirectImage_0" , ImageType::RenderImage2);
            image("indirectImage_1" , ImageType::RenderImage2);
            image("HDRImage_0" , ImageType::RenderImage2);
            image("HDRImage_1" , ImageType::RenderImage2);

            image("pixelCheck" , ImageType::RenderImage2);
            image("albedoImage" , ImageType::RenderImage2);
            image("filteredShadowImage" , ImageType::RenderImage2);
            image("filteredAmbientOcclusion" , ImageType::RenderImage2);
            // the ambient light of the sky
            image("skybox" , ImageType::Skybox2);
            image("normalImage_0" , ImageType::RenderImage2);
            image("normalImage_1" , ImageType::RenderImage2);
            // the froxel volume of the fog : 2D array
            image("fogImage" , ImageType::RenderImage2);

            buffer("generalFeatures");
            buffer("generalBuffer");
            buffer("fogData");

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [PostProcess]"});

            const uint8_t currentImage = inputData.context->sc->frameIndex % 2;

            const uint32_t x = (inputData.context->sc->extent.width + 7) / 8;
            const uint32_t y = (inputData.context->sc->extent.height + 7) / 8;

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

            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            return true;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindAttributes(sources);
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [PostProcess]"});

            const uint8_t id = inputData.context->sc->frameIndex % 2;

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 write = VK_ACCESS_2_SHADER_WRITE_BIT;
            constexpr VkAccessFlags2 uniformRead = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            std::array<VkImageMemoryBarrier2 , 12> imageBar;

            // the denoised light , the depth , the sky marks and the fog volume : only read
            imageBar[0] = image("diffuseImage" , id)->translateImage(read , stage);
            imageBar[1] = image("specularImage" , id)->translateImage(read , stage);
            imageBar[2] = image("depthImage" , id)->translateImage(read , stage);
            imageBar[3] = image("indirectImage" , id)->translateImage(read , stage);
            imageBar[4] = image("pixelCheck")->translateImage(read , stage);
            imageBar[5] = image("albedoImage")->translateImage(read , stage);
            imageBar[6] = image("fogImage")->translateImage(read , stage);

            // the result
            imageBar[7] = image("HDRImage" , id)->translateImage(write , stage);

            // the shadows and the occlusion
            imageBar[8] = image("filteredShadowImage")->translateImage(read , stage);
            imageBar[9] = image("filteredAmbientOcclusion")->translateImage(read , stage);

            // the ambient light of the sky
            imageBar[10] = skybox("skybox")->translateImage(read , stage);
            imageBar[11] = image("normalImage" , id)->translateImage(read , stage);

            std::array<VkBufferMemoryBarrier2 , 3> bufBar;

            bufBar[0] = buffer("generalFeatures")->translateBuffer(uniformRead , stage);
            bufBar[1] = buffer("generalBuffer")->translateBuffer(uniformRead , stage);
            bufBar[2] = buffer("fogData")->translateBuffer(uniformRead , stage);

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = static_cast<uint32_t>(bufBar.size());
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

        pShaderPipeType pipe{nullptr};

        CreateInfo inputData{};

        // samp of the shader : the fog volume is filtered
        pSampler sampler{nullptr};

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(PostProcess&& input) noexcept{
            if(this == &input) return;
            processDelete();
            GraphPass2::operator=(std::move(input));

            pipe = std::move(input.pipe);
            inputData = input.inputData;
            sampler = std::move(input.sampler);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete() noexcept{
            clearAttributes();
            // the images and the buffers belong to the graph , they are only forgotten
            pipe = nullptr;
            sampler = nullptr;

            inputData = {};
            completedInit = false;
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            VulkanContext* context = inputData.context;

            try{

                pipe = std::make_shared<ShaderPipeType>(context);

                {
                    Sampler::CreateInfo info{};
                    info.context = context;
                    info.linearClampSampler = true;

                    sampler = std::make_shared<Sampler>(info);
                }

                {
                    std::vector<DescriptorLayoutData> data(16);

                    // 0 .. 6 : diffuse , specular , pixelCheck , depth , indirect , albedo , HDR
                    for(uint32_t i = 0 ; i <= 6 ; i++){
                        data[i] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    }

                    data[7] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[8] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[9] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    // the fog volume and its sampler
                    data[10] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[11] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                    // the shadows and the occlusion
                    data[12] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[13] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    // the ambient light of the sky : the cube and the normals
                    data[14] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[15] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

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

                        std::vector<DescriptorData2> data(16);

                        data[0] = DescriptorData2::createData(image("diffuseImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0 , 0);
                        data[1] = DescriptorData2::createData(image("specularImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0);
                        data[2] = DescriptorData2::createData(image("pixelCheck") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0);
                        data[3] = DescriptorData2::createData(image("depthImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);
                        data[4] = DescriptorData2::createData(image("indirectImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 4 , 0);
                        data[5] = DescriptorData2::createData(image("albedoImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 5 , 0);
                        data[6] = DescriptorData2::createData(image("HDRImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 6 , 0);

                        data[7] = DescriptorData2::createData(buffer("generalFeatures") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 7 , 0);
                        data[8] = DescriptorData2::createData(buffer("generalBuffer") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 8 , 0);
                        data[9] = DescriptorData2::createData(buffer("fogData") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 9 , 0);

                        data[10] = DescriptorData2::createData(sampler.get() , 10 , 0);
                        data[11] = DescriptorData2::createData(image("fogImage") , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 11 , 0);

                        data[12] = DescriptorData2::createData(image("filteredShadowImage") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 12 , 0);
                        data[13] = DescriptorData2::createData(image("filteredAmbientOcclusion") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 13 , 0);

                        data[14] = DescriptorData2::createData(skybox("skybox") , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 14 , 0);
                        data[15] = DescriptorData2::createData(image("normalImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 15 , 0);

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
                    info.filePath = inputData.path;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create PostProcess pipe"});
            }

            completedInit = true;
            return {};
        }
    };

    using pPostProcess = std::shared_ptr<PostProcess>;
}
