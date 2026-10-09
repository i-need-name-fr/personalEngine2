#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define VMA_IMPLEMENTATION
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include "include/Buffer2/GeneralBuffer.hpp"
#include "include/Image2/ViewPerLayer.hpp"
#include "include/Image2/ViewPerMip.hpp"
#include "include/AS2/TLAS2.hpp"
#include "include/Object2/cube.hpp"

using namespace mox;

int main(){
    std::cout << "hello world\n";
    try{

        mox::VulkanContext::CreateInfo info{};
        info.asyncMultiThreadQueueCount = 4;
        info.multithreadQueueCount = 1;

        mox::VulkanContext context(info);

        std::vector<uint32_t> inputData{3974 , 8273, 1821, 982392 , 121, 129819, 283, 111 , 0, 4545};

        mox::GeneralBuffer2<mox::BufferType::Device , true>::CreateInfo data{};
        data.context = &context;
        data.size = sizeof(uint32_t) * 10;
        data.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        TextureManager2 manager{{&context}};
        mox::GeneralBuffer2<mox::BufferType::Device ,true> buffer(data , mox::translateToSpan<uint32_t>(inputData));


        pRenderImage2 image{nullptr};
        const uint32_t count = static_cast<uint32_t>(std::floor(std::log2(std::max(2560 , 1440)))) + 1;
        std::cout << std::format("mip count -> {}" , static_cast<uint32_t>(count)) << '\n';
        {
            RenderImage2::CreateInfo info{};
            info.clear.float32[0] = 0.0f;
            info.clear.float32[1] = 0.0f;
            info.clear.float32[2] = 0.0f;
            info.clear.float32[3] = 0.0f;
            info.context = &context;
            info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
            info.imageFormat = VK_FORMAT_R8G8B8A8_UNORM;
            info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            info.layerCount = 6;
            info.extent = {2560 , 1440};
            info.mipMapCount = count;
            image = std::make_shared<RenderImage2>(info);
        }
        {
            pViewPerLayer<5> views{nullptr};
            ViewPerLayer<5>::CreateInfo info{};
            info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            info.image = image->image;
            info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
            info.format = image->format;
            info.flags = 0;
            info.context = &context;

            views = std::make_shared<ViewPerLayer<5>>(info);
            std::cout << views->size() << '\n';
            if(views->at(5)) std::cout << "hello \n";
        }

        {
            pViewPerMip<11> views{nullptr};
            ViewPerMip<11>::CreateInfo info{};
            info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            info.image = image->image;
            info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
            info.format = image->format;
            info.flags = 0;
            info.context = &context;

            views = std::make_shared<ViewPerMip<11>>(info);
        }
        {
            // now lets test mesh

            Mesh2::CreateInfo info{};
            info.inputIndices = cube::indices;
            info.inputVertices = cube::vertices;
            info.roughness = 1.0f;
            info.metallic = 0.0f;
            info.roughnessPath = "3984983";
            info.needsCalcTBN = true;
            {
                const auto result = Mesh2::Begin(&context , &manager);
                if(!result.has_value()){
                    std::cerr << "failed to begin mesh\n";
                    return 1;
                }
            }

            Mesh2 mesh(info);

            {
                const auto result = Mesh2::End();
                if(!result.has_value()){
                    std::cerr << result.error().second << '\n';
                    return 1;
                }
            }

            mesh.addMatrix(glm::mat4(0.5f));
            Mesh2::updateMeshData();

            TLAS2 tlas(&context);
            std::cout << "BLAS count : " << Mesh2::BLAS_storage.size() << '\n';
            std::cout << "instance count : " << Mesh2::totalInstanceLayouts.size() << '\n';
            std::cout << "CMD count : " << Mesh2::totalCmdCount << '\n';

        }

        Mesh2::performDelete();
    }
    catch(std::exception &e){
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}