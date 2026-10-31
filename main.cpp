#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define VMA_IMPLEMENTATION
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define STB_IMAGE_IMPLEMENTATION
#define SPIRV_COMPILATION_REQUIRED

#include <iostream>
#include "Core2.hpp"

using namespace mox;

int main(){
    std::cout << "hello world\n";
    try{
        Core2 core(9);
        core.run();
        
    }
    catch(std::exception &e){
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}