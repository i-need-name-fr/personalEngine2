#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../Object2/Mesh2.hpp"

namespace mox{

    class AEHDR : public GraphPass2{
    public:

        AEHDR() = delete;
        AEHDR(AEHDR& input) = delete;
        AEHDR(const AEHDR& input) = delete;
        AEHDR& operator=(AEHDR& input) = delete;
        AEHDR& operator=(const AEHDR& input) = delete;

        AEHDR(AEHDR&& input) noexcept{
            processCopy(std::move(input));
        }

        AEHDR& operator=(AEHDR&& input) noexcept{
            processCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VulkanContext* context;
            std::string_view path;

        };

        explicit AEHDR(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager | VkReqTypePhysicalDevice , info.context);
            if(!result.has_value()){
                engineLogger(result.error());
                THROW_MESSAGE;
            }

            if(info.path.empty()){
                engineLogger({MOX_ERROR_TYPE_FAILED_CREATION , "failed to create AEHDR , since the path to file is empty"});
                THROW_MESSAGE;
            }

            context = info.context;
            pathToFile = std::string(info.path);
        }

        ~AEHDR(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{
            
            std::vector<GraphParameterInfo> reqs(3);
            reqs[0] ={ moxGraphTypeImage ,  ImageType::RenderImage2 ,"HDRImage_0"};
            reqs[1] ={ moxGraphTypeImage ,  ImageType::RenderImage2 ,"HDRImage_1"};

            reqs[2].name = "luminanceHistory";
            reqs[2].type = moxGraphTypeBuffer;

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [AEHDR]"});

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

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            return true;
        }

        [[nodiscard]] [[gnu::cold]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};
            const auto bound = bindAttributes(sources);
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{

            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to translate the buffers , since the AEHDR hasnt been created"});

            const uint8_t currentImage = context->sc->frameIndex %2;
            
            auto imageBar = image("HDRImage" , currentImage)->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            auto bufBar = buffer("luminanceHistory")->translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT ,VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT );

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.bufferMemoryBarrierCount = 1;
            info.imageMemoryBarrierCount = 0;
            info.pMemoryBarriers = nullptr;
            info.pBufferMemoryBarriers = &bufBar;
            info.memoryBarrierCount = 0;
            info.pNext = nullptr;
            info.dependencyFlags = 0;

            // we will translate the buffer first
            vkCmdPipelineBarrier2(cmd , &info);

            vkCmdFillBuffer(cmd , buffer("luminanceHistory")->buffer , 0 , buffer("luminanceHistory")->getBufferSize() , 0);

            bufBar = buffer("luminanceHistory")->translateBuffer(VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT );

            info.imageMemoryBarrierCount = 1;
            info.pImageMemoryBarriers = &imageBar;
            // now we translate buffer & image
            vkCmdPipelineBarrier2(cmd , &info);

            return {};
        }

    private:

        VulkanContext* context{nullptr};

        pShaderPipeType pipe{nullptr};

        std::string pathToFile{};

        void processCopy(AEHDR&& input) noexcept{
            if(this == &input) return;
            processDelete();
            GraphPass2::operator=(std::move(input));
            context = std::exchange(input.context, nullptr) ;
            pipe = std::move(input.pipe);
            completedInit = std::exchange(input.completedInit , false);
            pathToFile = std::move(input.pathToFile);
        }

        void processDelete() noexcept{
            clearAttributes();
            context = nullptr;
            completedInit = false;
            pipe = nullptr;
            pathToFile.clear();
        }

        [[nodiscard]] ErrorDataOutput<void> createPipe() noexcept{
            try{

                pipe = std::make_shared<ShaderPipeType>(context);
                // descriptor layout
                {
                    std::vector<DescriptorLayoutData> data(2);
                    data[0] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , .count = 1};
                    data[1] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                    pipe->descriptorLayouts.resize(1);
                    pipe->descriptorLayouts[0] = std::make_shared<DescriptorLayout>(info);
                }

                {
                    PipelineLayout::CreateInfo info{};
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.context = context;
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {
                    pipe->buffers.resize(2);
                    for(auto& id : {0 , 1}){
                        std::vector<DescriptorData2> data(2);
                        data[0] = DescriptorData2::createData(image("HDRImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  0 , 0);
                        data[1] = DescriptorData2::createData(buffer("luminanceHistory") , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 ,0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = pipe->descriptorLayouts[0].get();

                        pipe->buffers[id] = std::make_shared<DescriptorBuffer2>(info);
                    }
                }

                {
                    ShaderObjectType::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = pathToFile;
                    info.pushes = {};
                    info.layouts = {pipe->descriptorLayouts[0].get()};

                    pipe->shaders.resize(1);
                    pipe->shaders[0] = std::make_shared<ShaderObjectType>(info);
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create AEHDR , please check the logs above"});
            }
            completedInit = true;
            return {};
        }
    };

    using pAEHDR = std::shared_ptr<AEHDR>;
}