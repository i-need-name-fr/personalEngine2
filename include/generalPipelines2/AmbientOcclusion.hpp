#pragma once

#include <random>
#include "../ShaderPipe/ShaderPipe.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"

namespace mox{

    // SSAO / HBAO : reads the positions and the normals of the g buffer and the AO of the previous frame ( the history ) ,
    // writes the AO of this frame . the images ping-pong : the AO of frame N is written to ambientOcclusionImage[N % 2] and the one
    // that is read as the history is ambientOcclusionImage[( N + 1 ) % 2] . the kernel of the SSAO belongs to the pass
    class AmbientOcclusion : public GraphPass2{
    public:

        SetTypicalOperators(AmbientOcclusion)
        SetTypicalTransferOperators(AmbientOcclusion)

        struct CreateInfo{
            VulkanContext* context{nullptr};
            std::string path;
        };

        explicit AmbientOcclusion(const CreateInfo &info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
                if(!result.has_value()){
                    engineLogger(result);
                    THROW_MESSAGE;
                }
            }

            if(info.path.empty()){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create AmbientOcclusion , since the path to the shader is empty"});
                THROW_MESSAGE;
            }

            context = info.context;
            path = info.path;
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(8);
            reqs[0] = {moxGraphTypeImage , ImageType::RenderImage2 , "positionImage_0"};
            reqs[1] = {moxGraphTypeImage , ImageType::RenderImage2 , "positionImage_1"};
            reqs[2] = {moxGraphTypeImage , ImageType::RenderImage2 , "normalImage_0"};
            reqs[3] = {moxGraphTypeImage , ImageType::RenderImage2 , "normalImage_1"};
            reqs[4] = {moxGraphTypeImage , ImageType::RenderImage2 , "ambientOcclusionImage_0"};
            reqs[5] = {moxGraphTypeImage , ImageType::RenderImage2 , "ambientOcclusionImage_1"};

            reqs[6].type = moxGraphTypeBuffer;
            reqs[6].name = "generalBuffer";

            reqs[7].type = moxGraphTypeBuffer;
            reqs[7].name = "generalFeatures";

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to execute the pass , since the pipeline wasnt created yet [AmbientOcclusion]"});

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

                cmdBindDescriptorBuffersEXT(cmd , 1 , &info);

                uint32_t sets[1] = {0};
                VkDeviceSize offset[1] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offset);
            }

            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            // when the AO is not required the shader still writes 1.0 : the filter after it reads the image
            return true;
        }

        [[nodiscard]] [[gnu::cold]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            const auto bound = bindSources(sources ,
                GraphSlot<RenderImage2>{"positionImage_0" , &positionImage[0]} , GraphSlot<RenderImage2>{"positionImage_1" , &positionImage[1]} ,
                GraphSlot<RenderImage2>{"normalImage_0" , &normalImage[0]} , GraphSlot<RenderImage2>{"normalImage_1" , &normalImage[1]} ,
                GraphSlot<RenderImage2>{"ambientOcclusionImage_0" , &ambientOcclusionImage[0]} , GraphSlot<RenderImage2>{"ambientOcclusionImage_1" , &ambientOcclusionImage[1]} ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer} ,
                GraphSlot<BufferBase2>{"generalFeatures" , &features});
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [AmbientOcclusion]"});

            const uint8_t id = context->sc->frameIndex %2;

            std::array<VkImageMemoryBarrier2 , 4> imageBar;
            imageBar[0] = positionImage[id]->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[1] = normalImage[id]->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            // the history is the AO of the previous frame : it is only read
            imageBar[2] = ambientOcclusionImage[1 - id]->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            imageBar[3] = ambientOcclusionImage[id]->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

            std::array<VkBufferMemoryBarrier2 , 3> bufBar;
            bufBar[0] = generalBuffer->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            bufBar[1] = features->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            bufBar[2] = kernel->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

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

        // the size of the kernel that the shader reads ( it takes at most 64 samples )
        inline static constexpr uint32_t kernelSize = 64;

        VulkanContext* context{nullptr};
        std::string path;

        pShaderPipe pipe{nullptr};
        // device local only : the kernel is written once ( through a staging buffer ) and then only read by the shader
        pGeneralBuffer2<BufferType::Device , false> kernel{nullptr};

        RenderImage2* positionImage[2]{nullptr , nullptr};
        RenderImage2* normalImage[2]{nullptr , nullptr};
        RenderImage2* ambientOcclusionImage[2]{nullptr , nullptr};

        BufferBase2* generalBuffer{nullptr};
        BufferBase2* features{nullptr};

        void processCopy(AmbientOcclusion&& input) noexcept{
            if(this == &input) return;
            processDelete();

            context = std::exchange(input.context , nullptr);
            path = std::move(input.path);
            pipe = std::move(input.pipe);
            kernel = std::move(input.kernel);

            for(const auto i : {0 , 1}){
                positionImage[i] = std::exchange(input.positionImage[i] , nullptr);
                normalImage[i] = std::exchange(input.normalImage[i] , nullptr);
                ambientOcclusionImage[i] = std::exchange(input.ambientOcclusionImage[i] , nullptr);
            }

            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            features = std::exchange(input.features , nullptr);

            completedInit = std::exchange(input.completedInit , false);
        }

        void processDelete() noexcept{
            // the images and the buffers of the graph are only forgotten here , the kernel and the pipe are the ones of the pass
            pipe = nullptr;
            kernel = nullptr;
            context = nullptr;
            path.clear();

            for(const auto i : {0 , 1}){
                positionImage[i] = nullptr;
                normalImage[i] = nullptr;
                ambientOcclusionImage[i] = nullptr;
            }

            generalBuffer = nullptr;
            features = nullptr;
            completedInit = false;
        }

        // the hemisphere around the normal ( z > 0 in the tangent space ) : the samples are denser near the center , like in a typical SSAO ,
        // the seed is fixed , so the kernel is the same in every run
        [[nodiscard]] static std::vector<glm::vec4> generateKernel() noexcept{
            std::mt19937 generator(1337u);
            std::uniform_real_distribution<float> unit(0.0f , 1.0f);

            std::vector<glm::vec4> samples(kernelSize);
            for(uint32_t i = 0 ; i < kernelSize ; i++){
                glm::vec3 sample(unit(generator) * 2.0f - 1.0f , unit(generator) * 2.0f - 1.0f , unit(generator));
                sample = glm::normalize(sample) * unit(generator);

                float scale = static_cast<float>(i) / static_cast<float>(kernelSize);
                scale = glm::mix(0.1f , 1.0f , scale * scale);

                samples[i] = glm::vec4(sample * scale , 0.0f);
            }

            return samples;
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            try{
                // the kernel first : it is a descriptor of the set
                {
                    auto samples = generateKernel();

                    GeneralBuffer2<BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(glm::vec4) * kernelSize;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    kernel = std::make_shared<GeneralBuffer2<BufferType::Device , false>>(info , translateToSpan(samples));
                }

                pipe = std::make_shared<ShaderPipe>(context);

                {
                    // the order is the one of the bindings of the shader , binding 0 is the depth image that the shader declares but does not use
                    std::vector<DescriptorLayoutData> data(8);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};   // position
                    data[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};   // normal
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};   // history
                    data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};   // output
                    data[5] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};  // general buffer
                    data[6] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};  // features
                    data[7] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};  // kernel

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

                // the images ping-pong , so there are two descriptor buffers : the output of one is the history of the other
                for(const auto id : {0 , 1}){
                    std::vector<DescriptorData2> data(7);
                    data[0] = DescriptorData2::createData(positionImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0);
                    data[1] = DescriptorData2::createData(normalImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0);
                    data[2] = DescriptorData2::createData(ambientOcclusionImage[1 - id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);
                    data[3] = DescriptorData2::createData(ambientOcclusionImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 4 , 0);
                    data[4] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 5 , 0);
                    data[5] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 6 , 0);
                    data[6] = DescriptorData2::createData(kernel.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 7 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->addDescriptorBuffer2(info);
                }

                {
                    ShaderObject::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = path;
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.pushes = {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception& e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the AmbientOcclusion pipeline , please check the logs above"});
            }

            completedInit = true;
            return {};
        }
    };

    using pAmbientOcclusion = std::shared_ptr<AmbientOcclusion>;
}
