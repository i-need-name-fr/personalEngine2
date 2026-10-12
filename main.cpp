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
#include "include/Image2/FrameTargets.hpp"

#include "include/AS2/TLAS2.hpp"
#include "include/Object2/cube.hpp"

#include "include/Image2/MultiViewImage2.hpp"

using namespace mox;

int main(){
    std::cout << "hello world\n";
    try{

    }
    catch(std::exception &e){
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}