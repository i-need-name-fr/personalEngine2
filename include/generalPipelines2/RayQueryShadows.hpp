#pragma once

#include "../ShaderPipe2/ShaderPipeSelect.hpp"
#include "../GraphRender2/GraphRender2.hpp"
#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../AS2/AccelerationStructureBase.hpp"
#include "../Object2/Mesh2.hpp"
#include "../Image2/TextureManager2.hpp"

namespace mox{

    class RayQueryShadows : public GraphPass2{
    public:

        SetTypicalOperators(RayQueryShadows)
        SetTypicalTransferOperators(RayQueryShadows)

        struct CreateInfo{
            VulkanContext* context;
            AccelerationStructure* as{};
            // the textures : the alpha test of the leaves reads the base colour map ( set 1 )
            TextureManager2* manager{};
            std::string path;
        };

        explicit RayQueryShadows(const CreateInfo &info){
            const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice |VkReqTypeQueueManager , info.context);
            if(!result.has_value()){
                engineLogger(result);
                THROW_MESSAGE;
            }

            if(!info.manager){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create RayQueryShadows , since the texture manager is nullptr ( the alpha test needs the textures )"});
                THROW_MESSAGE;
            }

            inputData = info;
        }

        ~RayQueryShadows()noexcept{
            processDelete();
        }

        std::vector<GraphParameterInfo> acquireRequirements() noexcept override final{

            std::vector<GraphParameterInfo> reqs(16);

            reqs[0].type = moxGraphTypeBuffer;
            reqs[0].name = "generalBuffer";

            // the features are read by the CPU in requestExecution , so the post process checks that this buffer is a DeviceReadableBuffer
            reqs[1].type = moxGraphTypeBuffer;
            reqs[1].name = "generalFeatures";

            reqs[2].type = moxGraphTypeBuffer;
            reqs[2].name = "sunBuffer";

            reqs[3].type = moxGraphTypeBuffer;
            reqs[3].name = "lightBuffer";

            reqs[4].type = moxGraphTypeImage;
            reqs[4].name = "shadowImage_0";
            reqs[4].imageSpecification.type = ImageType::RenderImage2;

            reqs[5].type = moxGraphTypeImage;
            reqs[5].name = "shadowImage_1";
            reqs[5].imageSpecification.type = ImageType::RenderImage2;

            reqs[6].type = moxGraphTypeImage;
            reqs[6].name = "positionImage_0";
            reqs[6].imageSpecification.type = ImageType::RenderImage2;

            reqs[7].type = moxGraphTypeImage;
            reqs[7].name = "positionImage_1";
            reqs[7].imageSpecification.type = ImageType::RenderImage2;

            reqs[8].type = moxGraphTypeImage;
            reqs[8].name = "normalImage_0";
            reqs[8].imageSpecification.type = ImageType::RenderImage2;

            reqs[9].type = moxGraphTypeImage;
            reqs[9].name = "normalImage_1";
            reqs[9].imageSpecification.type = ImageType::RenderImage2;

            reqs[10].type = moxGraphTypeImage;
            reqs[10].name = "pixelCheck";
            reqs[10].imageSpecification.type = ImageType::RenderImage2;

            // the lamps of the tiles ( Forward+ ) : the shadow is made from all the lamps of the tile of the pixel
            reqs[11].type = moxGraphTypeBuffer;
            reqs[11].name = "tileLightIndices";

            reqs[12].type = moxGraphTypeBuffer;
            reqs[12].name = "lightCountPerTile";

            reqs[13].type = moxGraphTypeBuffer;
            reqs[13].name = "forwardInfo";

            // the indirect light of the last frame : its luminance fills the shadows ( the pass runs before the indirect pass of this frame )
            reqs[14].type = moxGraphTypeImage;
            reqs[14].name = "indirectImage_0";
            reqs[14].imageSpecification.type = ImageType::RenderImage2;

            reqs[15].type = moxGraphTypeImage;
            reqs[15].name = "indirectImage_1";
            reqs[15].imageSpecification.type = ImageType::RenderImage2;

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
                std::array<VkDescriptorBufferBindingInfoEXT , 2> info{};
                info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[0].address = pipe->buffers[currentImage]->pBuffer;
                info[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[0].pNext = nullptr;

                // set 1 : the textures of the TextureManager2
                info[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info[1].address = inputData.manager->dbuffer->pBuffer;
                info[1].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                info[1].pNext = nullptr;

                cmdBindDescriptorBuffersEXT(cmd , 2 ,  info.data());

                uint32_t sets[2] = {0 , 1};
                VkDeviceSize offset[2] = {0 , 0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 2 , sets , offset);
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

            const auto bound = bindAttributes(sources);
            if(!bound.has_value()) return bound;

            // the features are read by the CPU ( requestExecution ) : the buffer has to be a DeviceReadableBuffer . bit_cast does not check the type , it takes
            // the address as it is ( the only base of the buffer is BufferBase2 , so the address of the base and of the buffer are the same ) :
            // the buffer of another type would be read as garbage , so the real type is checked first
            if(!dynamic_cast<DeviceReadableBuffer*>(buffer("generalFeatures"))){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create RayQueryShadows , since the generalFeatures buffer is not a DeviceReadableBuffer ( the CPU has to read it )"});
            }
            features = std::bit_cast<DeviceReadableBuffer*>(buffer("generalFeatures"));

            return createPipe();
        }

        [[nodiscard]] [[gnu::hot]] const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept override final{
            const uint8_t currentImage = inputData.context->sc->frameIndex %2;

            std::array<VkImageMemoryBarrier2 , 6> images;

            images[0] = image("shadowImage" , currentImage)->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            images[1] = image("shadowImage" , !currentImage)->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            images[2] = image("pixelCheck")->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            images[3] = image("positionImage" , currentImage)->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            images[4] = image("normalImage" , currentImage)->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            // the indirect light of the last frame
            images[5] = image("indirectImage" , !currentImage)->translateImage(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);

            std::array<VkBufferMemoryBarrier2 , 12> buffers;

            buffers[0] = buffer("generalBuffer")->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[1] = features->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[2] = buffer("sunBuffer")->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[3] = buffer("lightBuffer")->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            // the instances and the materials : the shader looks at the candidates of the ray
            buffers[4] = Mesh2::instanceLayoutBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[5] = Mesh2::LUT_buffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            // the lists of the lamps of the tiles : written by the ForwardPlus
            buffers[6] = buffer("tileLightIndices")->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[7] = buffer("lightCountPerTile")->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[8] = buffer("forwardInfo")->translateBuffer(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            // the geometry of the candidates : the UV for the alpha test
            buffers[9] = Mesh2::vertexBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[10] = Mesh2::indicesBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);
            buffers[11] = Mesh2::MeshBuffer->translateBuffer(VK_ACCESS_2_SHADER_READ_BIT_KHR , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR);

            auto dep = TypicalDependency<12 , 6>(images.data() , buffers.data());

            vkCmdPipelineBarrier2(cmd , &dep);

            return {};
        }

    private:

        pShaderPipeType pipe{nullptr};
        CreateInfo inputData{};

        DeviceReadableBuffer* features{nullptr}; // this one will be like this , since we will need to read it

        [[nodiscard]] ErrorDataOutput<void> createPipe(){
            VulkanContext* context = inputData.context;
            try{

                pipe = std::make_shared<ShaderPipeType>(context);
                {
                    std::vector<DescriptorLayoutData> data(20);
                    data[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    // 3 was the reservoirs of ReSTIR DI : reserved ( no descriptor ) , the other bindings keep their numbers
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0};
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

                    // the geometry of the candidates ( the alpha test ) : the vertices , the indices , the meshes
                    data[17] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[18] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[19] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

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
                    info.layouts = {pipe->descriptorLayouts[0].get() , inputData.manager->layout.get()};
                    info.context = context;
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {
                    for(auto id : {0  ,1}){
                        std::vector<DescriptorData2> data(20);
                        data[0] = DescriptorData2::createData(buffer("generalBuffer") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                        data[1] = DescriptorData2::createData(features , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0);
                        data[2] = DescriptorData2::createData(buffer("sunBuffer") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0);

                        data[4] = DescriptorData2::createData(buffer("lightBuffer") , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0);

                        data[5] = DescriptorData2::createData(inputData.as  , 5 , 0);

                        data[6] = DescriptorData2::createData(image("shadowImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 6 , 0);
                        data[7] = DescriptorData2::createData(image("shadowImage" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 7 , 0);
                        data[8] = DescriptorData2::createData(image("pixelCheck") , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0);
                        data[9] = DescriptorData2::createData(image("positionImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 9 , 0);
                        data[10] = DescriptorData2::createData(image("normalImage" , id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 10 , 0);

                        data[11] = DescriptorData2::createData(Mesh2::instanceLayoutBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 11 , 0);
                        data[12] = DescriptorData2::createData(Mesh2::LUT_buffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 12 , 0);

                        data[13] = DescriptorData2::createData(buffer("tileLightIndices") , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 13 , 0);
                        data[14] = DescriptorData2::createData(buffer("lightCountPerTile") , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 14 , 0);
                        data[15] = DescriptorData2::createData(buffer("forwardInfo") , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 15 , 0);

                        // the indirect light of the frame before this one ( the same parity rule as the shadow image of the last frame )
                        data[16] = DescriptorData2::createData(image("indirectImage" , !id) , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 16 , 0);

                        data[17] = DescriptorData2::createData(Mesh2::vertexBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 17 , 0);
                        data[18] = DescriptorData2::createData(Mesh2::indicesBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 18 , 0);
                        data[19] = DescriptorData2::createData(Mesh2::MeshBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 19 , 0);

                        // the binding 3 is reserved ( it was ReSTIR DI ) : nothing is written there
                        data.erase(data.begin() + 3);

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
                    info.layouts = {pipe->descriptorLayouts[0].get() , inputData.manager->layout.get()};
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
            GraphPass2::operator=(std::move(input));

            pipe = std::move(input.pipe);
            // the acceleration structure is not owned by the pass
            inputData = input.inputData;

            // the buffers and the images of the graph
            features = std::exchange(input.features , nullptr);

            // the moved from pass has no pipe any more : it is not initialized
            completedInit = std::exchange(input.completedInit , false);
            input.inputData = {};
        }

        void processDelete(void) noexcept{
            clearAttributes();
            // the buffers , the images and the acceleration structure belong to the graph or to the owner , they are only forgotten
            pipe = nullptr;

            features = nullptr;

            inputData = {};
            completedInit = false;
        }
    };

    using pRayQueryShadows = std::shared_ptr<RayQueryShadows>;
}
