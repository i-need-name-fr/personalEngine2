#pragma once

#include "../ShaderPipe/ShaderPipe.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Image2/TextureManager2.hpp"
#include "../Object2/Mesh2.hpp"

/*

[[vk::binding(0 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(1 , 0)]] ConstantBuffer<FogVolume> fogData;
[[vk::binding(2 , 0)]] ConstantBuffer<SunSource> sunBuffer;
[[vk::binding(3 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardBuffer;

[[vk::binding(4 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(5 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(6 , 0)]] RWStructuredBuffer<LightSource> lightBuffer;


[[vk::image_format("rgba16f")]]
[[vk::binding(7 , 0)]] RWTexture2DArray<float4> fogImage;
[[vk::image_format("r32f")]]
[[vk::binding(8 , 0)]] RWTexture2D<float> filteredShadowImage;

*/

namespace mox{

    class VolumentricRender : public GraphPass2{
    public:

        SetTypicalOperators(VolumentricRender)
        SetTypicalTransferOperators(VolumentricRender)

        struct CreateInfo{
            VulkanContext* context;
            std::string path;
        };

        explicit VolumentricRender(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~VolumentricRender(){
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> req(9);

            req[0].name = "generalBuffer";
            req[0].type = moxGraphTypeBuffer;

            req[1].name = "fogData";
            req[1].type = moxGraphTypeBuffer;

            req[2].name = "sunBuffer";
            req[2].type = moxGraphTypeBuffer;

            req[3].name = "forwardInfo";
            req[3].type = moxGraphTypeBuffer;

            req[4].name = "tileLightIndices";
            req[4].type = moxGraphTypeBuffer;

            req[5].name = "lightCountPerTile";
            req[5].type = moxGraphTypeBuffer;

            req[6].name = "lightBuffer";
            req[6].type = moxGraphTypeBuffer;

            req[7].name = "fogImage";
            req[7].type = moxGraphTypeImage;
            req[7].imageSpecification.type = ImageType::RenderImage2;

            req[8].name = "filteredShadowImage";
            req[8].type = moxGraphTypeImage;
            req[8].imageSpecification.type = ImageType::RenderImage2;

            return req;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [VolumentricRender]"});

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.address = pipe->buffers[0]->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 1,  &info);

                uint32_t sets[1] = {0};
                VkDeviceSize offset[] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , sets , offset);
            }
            // the groups are 4 x 4 , the froxel volume has 64 layers ( z )
            const uint32_t x = (fogImage->extent.width + 3) / 4;
            const uint32_t y = (fogImage->extent.height + 3) / 4;

            vkCmdDispatch(cmd , x , y, 64);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            return true;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindSources(sources ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer} , GraphSlot<BufferBase2>{"fogData" , &fogData} ,
                GraphSlot<BufferBase2>{"sunBuffer" , &sunBuffer} , GraphSlot<BufferBase2>{"forwardInfo" , &forwardInfo} ,
                GraphSlot<BufferBase2>{"tileLightIndices" , &tileLightIndices} , GraphSlot<BufferBase2>{"lightCountPerTile" , &lightCountPerTile} ,
                GraphSlot<BufferBase2>{"lightBuffer" , &lightBuffer} ,
                GraphSlot<RenderImage2>{"fogImage" , &fogImage} , GraphSlot<RenderImage2>{"filteredShadowImage" , &filteredShadowImage});
            if(!bound.has_value()) return bound;

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate the barriers , since the pipeline wasnt created yet [VolumentricRender]"});

            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            constexpr VkAccessFlags2 uniformRead = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;

            // everything is only read here : the uniform buffers and the lights that the ForwardPlus has just written ( the shader declares
            // the lists as RW , but it does not write them )
            std::array<VkBufferMemoryBarrier2 , 7> bufBar;
            bufBar[0] = generalBuffer->translateBuffer(uniformRead , stage);
            bufBar[1] = fogData->translateBuffer(uniformRead , stage);
            bufBar[2] = sunBuffer->translateBuffer(uniformRead , stage);
            bufBar[3] = forwardInfo->translateBuffer(uniformRead , stage);
            bufBar[4] = tileLightIndices->translateBuffer(read , stage);
            bufBar[5] = lightCountPerTile->translateBuffer(read , stage);
            bufBar[6] = lightBuffer->translateBuffer(read , stage);

            std::array<VkImageMemoryBarrier2 , 2> imageBar;
            // the froxel volume is written , the filtered shadow is read
            imageBar[0] = fogImage->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , stage);
            imageBar[1] = filteredShadowImage->translateImage(read , stage);

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
        
        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(VolumentricRender&& input) noexcept{
            if(this == &input) return;
            processDelete();

            pipe = std::move(input.pipe);
            inputData = input.inputData;

            // the buffers and the images of the graph
            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            fogData = std::exchange(input.fogData , nullptr);
            sunBuffer = std::exchange(input.sunBuffer , nullptr);
            forwardInfo = std::exchange(input.forwardInfo , nullptr);
            tileLightIndices = std::exchange(input.tileLightIndices , nullptr);
            lightCountPerTile = std::exchange(input.lightCountPerTile , nullptr);
            lightBuffer = std::exchange(input.lightBuffer , nullptr);
            fogImage = std::exchange(input.fogImage , nullptr);
            filteredShadowImage = std::exchange(input.filteredShadowImage , nullptr);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete(void) noexcept{
            // the buffers and the images belong to the graph , they are only forgotten
            pipe = nullptr;

            generalBuffer = nullptr;
            fogData = nullptr;
            sunBuffer = nullptr;
            forwardInfo = nullptr;
            tileLightIndices = nullptr;
            lightCountPerTile = nullptr;
            lightBuffer = nullptr;
            fogImage = nullptr;
            filteredShadowImage = nullptr;

            inputData = {};
            completedInit = false;
        }

        [[nodiscard]] ErrorDataOutput<void > createPipe() noexcept{

            VulkanContext* context = inputData.context;

            try{

                pipe = std::make_shared<ShaderPipe>(inputData.context);

                {
                    std::vector<DescriptorLayoutData> data(9);
                    data[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[5] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[6] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    data[7] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[8] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context=  context;
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
                    std::vector<DescriptorData2> data(9);
                    data[0] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                    data[1] = DescriptorData2::createData(fogData , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1, 0);
                    data[2] = DescriptorData2::createData(sunBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0);
                    data[3] = DescriptorData2::createData(forwardInfo , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 3 , 0);

                    data[4] = DescriptorData2::createData(tileLightIndices , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0);
                    data[5] = DescriptorData2::createData(lightCountPerTile , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 5 , 0);
                    data[6] = DescriptorData2::createData(lightBuffer , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 6 , 0);

                    data[7] = DescriptorData2::createData(fogImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 7 , 0);
                    data[8] = DescriptorData2::createData(filteredShadowImage , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = {pipe->descriptorLayouts[0].get()};

                    pipe->addDescriptorBuffer2(info);
                }

                {
                    ShaderObject::CreateInfo info{};
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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Volumentric render"});
            }

            completedInit = true;
            return {};
        }

        pShaderPipe pipe{nullptr};

        CreateInfo inputData{};

        BufferBase2* generalBuffer{nullptr};
        BufferBase2* fogData{nullptr};
        BufferBase2* sunBuffer{nullptr};

        BufferBase2* forwardInfo{nullptr};
        BufferBase2* tileLightIndices{nullptr};
        BufferBase2* lightCountPerTile{nullptr};
        BufferBase2* lightBuffer{nullptr};

        // the froxel volume : 160 x 90 pixels , 64 layers ( a 2D array )
        RenderImage2* fogImage{nullptr};
        RenderImage2* filteredShadowImage{nullptr};
    };

    using pVolumentricRender = std::shared_ptr<VolumentricRender>;
}