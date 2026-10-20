#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../AS2/AccelerationStructureBase.hpp"
#include "../Object2/Mesh2.hpp"

namespace mox{

    class RayQueryShadows : public GraphPass2{
    public:

        SetTypicalOperators(RayQueryShadows)
        SetTypicalTransferOperators(RayQueryShadows)

        struct CreateInfo{
            VulkanContext* context;
            AccelerationStructure* as{};
            std::string path;
        };

        explicit RayQueryShadows(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice |VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~RayQueryShadows()noexcept{
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(18);

            reqs[0].type = moxGraphTypeBuffer;
            reqs[0].name = "generalBuffer";

            // the features are read by the CPU in requestExecution , so the post process checks that this buffer is a DeviceReadableBuffer
            reqs[1].type = moxGraphTypeBuffer;
            reqs[1].name = "generalFeatures";

            reqs[2].type = moxGraphTypeBuffer;
            reqs[2].name = "sunBuffer";

            // the reservoirs of ReSTIR DI : one per frame ( the shader reads the one that was written in this frame )
            reqs[3].type = moxGraphTypeBuffer;
            reqs[3].name = "ReSTIRDIBuffer_0";

            reqs[4].type = moxGraphTypeBuffer;
            reqs[4].name = "ReSTIRDIBuffer_1";

            reqs[5].type = moxGraphTypeBuffer;
            reqs[5].name = "lightBuffer";

            reqs[6].type = moxGraphTypeImage;
            reqs[6].name = "shadowImage_0";
            reqs[6].imageSpecification.type = ImageType::RenderImage2;

            reqs[7].type = moxGraphTypeImage;
            reqs[7].name = "shadowImage_1";
            reqs[7].imageSpecification.type = ImageType::RenderImage2;

            reqs[8].type = moxGraphTypeImage;
            reqs[8].name = "positionImage_0";
            reqs[8].imageSpecification.type = ImageType::RenderImage2;

            reqs[9].type = moxGraphTypeImage;
            reqs[9].name = "positionImage_1";
            reqs[9].imageSpecification.type = ImageType::RenderImage2;

            reqs[10].type = moxGraphTypeImage;
            reqs[10].name = "normalImage_0";
            reqs[10].imageSpecification.type = ImageType::RenderImage2;

            reqs[11].type = moxGraphTypeImage;
            reqs[11].name = "normalImage_1";
            reqs[11].imageSpecification.type = ImageType::RenderImage2;

            reqs[12].type = moxGraphTypeImage;
            reqs[12].name = "pixelCheck";
            reqs[12].imageSpecification.type = ImageType::RenderImage2;

            // the lamps of the tiles ( Forward+ ) : the shadow is made from all the lamps of the tile of the pixel
            reqs[13].type = moxGraphTypeBuffer;
            reqs[13].name = "tileLightIndices";

            reqs[14].type = moxGraphTypeBuffer;
            reqs[14].name = "lightCountPerTile";

            reqs[15].type = moxGraphTypeBuffer;
            reqs[15].name = "forwardInfo";

            // the indirect light of the last frame : its luminance fills the shadows ( the pass runs before the indirect pass of this frame )
            reqs[16].type = moxGraphTypeImage;
            reqs[16].name = "indirectImage_0";
            reqs[16].imageSpecification.type = ImageType::RenderImage2;

            reqs[17].type = moxGraphTypeImage;
            reqs[17].name = "indirectImage_1";
            reqs[17].imageSpecification.type = ImageType::RenderImage2;

            return reqs;
        }

        [[nodiscard]] [[gnu::hot]] ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept override final{
            [[unlikely]] if(!completedInit) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to execute the pass ,since the pipeline wasnt created yet [RayQueryShadows]"});

            const uint8_t currentImage = inputData.context->sc->frameIndex %2;

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

            vkCmdDispatch(cmd , x , y, 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            return {};
        }

        [[nodiscard]] [[gnu::hot]] const bool requestExecution() const noexcept override final{
            auto data = features->getBuffetDataSingular<GeneralFeatures>();
            const bool checkUp = !data.PT  && data.RayQuery && !data.VSM;
            return checkUp;
        }

        [[nodiscard]] const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept override final{
            if(completedInit) return {};

            const auto bound = bindSources(sources ,
                GraphSlot<BufferBase2>{"generalBuffer" , &generalBuffer} , GraphSlot<BufferBase2>{"generalFeatures" , &rawFeaturePtr} ,
                GraphSlot<BufferBase2>{"sunBuffer" , &sunBuffer} ,
                GraphSlot<BufferBase2>{"ReSTIRDIBuffer_0" , &ReSTIRDIBuffer[0]} , GraphSlot<BufferBase2>{"ReSTIRDIBuffer_1" , &ReSTIRDIBuffer[1]} ,
                GraphSlot<BufferBase2>{"lightBuffer" , &lightBuffer} ,
                GraphSlot<RenderImage2>{"shadowImage_0" , &shadowImage[0]} , GraphSlot<RenderImage2>{"shadowImage_1" , &shadowImage[1]} ,
                GraphSlot<RenderImage2>{"positionImage_0" , &positionImage[0]} , GraphSlot<RenderImage2>{"positionImage_1" , &positionImage[1]} ,
                GraphSlot<RenderImage2>{"normalImage_0" , &normalImage[0]} , GraphSlot<RenderImage2>{"normalImage_1" , &normalImage[1]} ,
                GraphSlot<RenderImage2>{"pixelCheck" , &pixelCheck} ,
                GraphSlot<BufferBase2>{"tileLightIndices" , &tileLightIndices} , GraphSlot<BufferBase2>{"lightCountPerTile" , &lightCountPerTile} ,
                GraphSlot<BufferBase2>{"forwardInfo" , &forwardInfo} ,
                GraphSlot<RenderImage2>{"indirectImage_0" , &indirectImage[0]} , GraphSlot<RenderImage2>{"indirectImage_1" , &indirectImage[1]});
            if(!bound.has_value()) return bound;

            // the features are read by the CPU ( requestExecution ) : the buffer has to be a DeviceReadableBuffer . bit_cast does not check the type , it takes
            // the address as it is ( the only base of the buffer is BufferBase2 , so the address of the base and of the buffer are the same ) :
            // the buffer of another type would be read as garbage , so the real type is checked first
            if(!dynamic_cast<DeviceReadableBuffer*>(rawFeaturePtr)){
                rawFeaturePtr = nullptr;
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create RayQueryShadows , since the generalFeatures buffer is not a DeviceReadableBuffer ( the CPU has to read it )"});
            }
            features = std::bit_cast<DeviceReadableBuffer*>(rawFeaturePtr);

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            const uint8_t currentImage = inputData.context->sc->frameIndex %2;

            std::array<VkImageMemoryBarrier2 , 6> images;

            images[0] = shadowImage[currentImage]->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            images[1] = shadowImage[!currentImage]->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            images[2] = pixelCheck->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            images[3] = positionImage[currentImage]->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            images[4] = normalImage[currentImage]->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            // the indirect light of the last frame
            images[5] = indirectImage[!currentImage]->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);

            std::array<VkBufferMemoryBarrier2 , 10> buffers;

            buffers[0] = generalBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[1] = features->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[2] = sunBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[3] = ReSTIRDIBuffer[currentImage]->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[4] = lightBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            // the instances and the materials : the shader looks at the candidates of the ray
            buffers[5] = Mesh2::instanceLayoutBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[6] = Mesh2::LUT_buffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            // the lists of the lamps of the tiles : written by the ForwardPlus
            buffers[7] = tileLightIndices->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[8] = lightCountPerTile->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[9] = forwardInfo->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);

            auto dep = TypicalDependency<10 , 6>(images.data() , buffers.data());

            vkCmdPipelineBarrier2(cmd , &dep);

            return {};
        }


    private:

        pShaderPipeType pipe{nullptr};
        CreateInfo inputData{};

        BufferBase2* generalBuffer{nullptr};
        DeviceReadableBuffer* features{nullptr}; // this one will be like this , since we will need to read it
        BufferBase2* rawFeaturePtr{nullptr}; // this one we will enter to post process
        BufferBase2* sunBuffer{nullptr};
        BufferBase2* ReSTIRDIBuffer[2]{nullptr , nullptr};
        BufferBase2* lightBuffer{nullptr};
        BufferBase2* tileLightIndices{nullptr};
        BufferBase2* lightCountPerTile{nullptr};
        BufferBase2* forwardInfo{nullptr};

        RenderImage2* shadowImage[2]{nullptr , nullptr};
        RenderImage2* positionImage[2]{nullptr , nullptr};
        RenderImage2* normalImage[2]{nullptr , nullptr};
        RenderImage2* indirectImage[2]{nullptr , nullptr};

        RenderImage2* pixelCheck{nullptr};

        [[nodiscard]] ErrorDataOutput<void> createPipe(){
            VulkanContext* context = inputData.context;
            try{

                pipe = std::make_shared<ShaderPipeType>(context);
                {
                    std::vector<DescriptorLayoutData> data(17);
                    data[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    data[5] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR , 1};

                    data[6] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[7] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[8] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[9] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[10] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    // the instances and the materials of Mesh2
                    data[11] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[12] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    // the lamps of the tiles
                    data[13] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[14] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[15] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    // the indirect light of the last frame
                    data[16] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

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
                    for(auto id : {0  ,1}){
                        std::vector<DescriptorData2> data(17);
                        data[0] = DescriptorData2::createData(generalBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                        data[1] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0);
                        data[2] = DescriptorData2::createData(sunBuffer , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0);

                        data[3] = DescriptorData2::createData(ReSTIRDIBuffer[!id] , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0);
                        data[4] = DescriptorData2::createData(lightBuffer , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0);

                        data[5] = DescriptorData2::createData(inputData.as  , 5 , 0);

                        data[6] = DescriptorData2::createData(shadowImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 6 , 0);
                        data[7] = DescriptorData2::createData(shadowImage[!id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 7 , 0);
                        data[8] = DescriptorData2::createData(pixelCheck , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0);
                        data[9] = DescriptorData2::createData(positionImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 9 , 0);
                        data[10] = DescriptorData2::createData(normalImage[id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 10 , 0);

                        data[11] = DescriptorData2::createData(Mesh2::instanceLayoutBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 11 , 0);
                        data[12] = DescriptorData2::createData(Mesh2::LUT_buffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 12 , 0);

                        data[13] = DescriptorData2::createData(tileLightIndices , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 13 , 0);
                        data[14] = DescriptorData2::createData(lightCountPerTile , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 14 , 0);
                        data[15] = DescriptorData2::createData(forwardInfo , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 15 , 0);

                        // the indirect light of the frame before this one ( the same parity rule as the shadow image of the last frame )
                        data[16] = DescriptorData2::createData(indirectImage[!id] , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 16 , 0);

                        DescriptorBuffer2::CreateInfo info{};
                        info.context = context;
                        info.data = data;
                        info.layout = {pipe->descriptorLayouts[0].get()};

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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RayQueryShadows , please check the logs above"});
            }

            completedInit = true;
            return {};
        }

        // the macro of the move operators already checks this == &input , the check here keeps the function safe on its own
        void processCopy(RayQueryShadows&& input) noexcept{
            if(this == &input) return;
            processDelete();

            pipe = std::move(input.pipe);
            // the acceleration structure is not owned by the pass
            inputData = input.inputData;

            // the buffers and the images of the graph
            generalBuffer = std::exchange(input.generalBuffer , nullptr);
            features = std::exchange(input.features , nullptr);
            rawFeaturePtr = std::exchange(input.rawFeaturePtr , nullptr);
            sunBuffer = std::exchange(input.sunBuffer , nullptr);
            lightBuffer = std::exchange(input.lightBuffer , nullptr);
            tileLightIndices = std::exchange(input.tileLightIndices , nullptr);
            lightCountPerTile = std::exchange(input.lightCountPerTile , nullptr);
            forwardInfo = std::exchange(input.forwardInfo , nullptr);

            for(const auto i : {0 , 1}){
                ReSTIRDIBuffer[i] = std::exchange(input.ReSTIRDIBuffer[i] , nullptr);
                shadowImage[i] = std::exchange(input.shadowImage[i] , nullptr);
                positionImage[i] = std::exchange(input.positionImage[i] , nullptr);
                normalImage[i] = std::exchange(input.normalImage[i] , nullptr);
                indirectImage[i] = std::exchange(input.indirectImage[i] , nullptr);
            }
            pixelCheck = std::exchange(input.pixelCheck , nullptr);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete(void) noexcept{
            // the buffers , the images and the acceleration structure belong to the graph or to the owner , they are only forgotten
            pipe = nullptr;

            generalBuffer = nullptr;
            features = nullptr;
            rawFeaturePtr = nullptr;
            sunBuffer = nullptr;
            lightBuffer = nullptr;
            tileLightIndices = nullptr;
            lightCountPerTile = nullptr;
            forwardInfo = nullptr;

            for(const auto i : {0 , 1}){
                ReSTIRDIBuffer[i] = nullptr;
                shadowImage[i] = nullptr;
                positionImage[i] = nullptr;
                normalImage[i] = nullptr;
                indirectImage[i] = nullptr;
            }
            pixelCheck = nullptr;

            inputData = {};
            completedInit = false;
        }
    };

    using pRayQueryShadows = std::shared_ptr<RayQueryShadows>;
}
