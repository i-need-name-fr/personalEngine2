#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include "../logger/logger.hpp"
#include <utility>

namespace mox{

    #ifdef NDEBUG
        const bool enableValidations = false;
    #else
        const bool enableValidations = true;
    #endif

    #define THROW_MESSAGE throw std::runtime_error("error occured , check logs")

    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    // device extensions
    inline static const std::vector<const char*> requiredExt = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        // promoted to core in Vulkan 1.2, but requesting it explicitly keeps
        // checkExtensions() honest about what the renderer actually relies on
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,    // ОБЯЗАТЕЛЬНО тоже, раз без pipeline
        VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME  ,
        VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME
    };


    inline void vkCheck(VkResult result , const char* what){
        if(result != VK_SUCCESS){
            throw std::runtime_error(std::string(what) + " failed with VkResult " + std::to_string(static_cast<int>(result)));
        }
    }
    #define VK_CHECK(call) vkCheck((call) , #call)

    inline PFN_vkCreateAccelerationStructureKHR createAccelerationStructureKHR = nullptr;
    inline PFN_vkGetAccelerationStructureBuildSizesKHR getAccelerationStructureBuildSizesKHR = nullptr;
    inline PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructuresKHR = nullptr;
    inline PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructureKHR = nullptr;
    inline PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelerationStructureDeviceAddressKHR = nullptr;

    inline PFN_vkCreateRayTracingPipelinesKHR createRayTracingPipelinesKHR = nullptr;
    inline PFN_vkGetRayTracingShaderGroupHandlesKHR getRayTracingShaderGroupHandlesKHR = nullptr;
    inline PFN_vkCmdTraceRaysKHR cmdTraceRaysKHR = nullptr;
    
    inline PFN_vkCreateShadersEXT createShaderEXT = nullptr;
    inline PFN_vkCmdSetVertexInputEXT cmdSetVertexInputEXT = nullptr;
    // shader objects make every pipeline state dynamic , and these four have no
    // core equivalent - without them the state stays undefined at draw time
    inline PFN_vkCmdSetPolygonModeEXT cmdSetPolygonModeEXT = nullptr;
    inline PFN_vkCmdSetRasterizationSamplesEXT cmdSetRasterizationSamplesEXT = nullptr;
    inline PFN_vkCmdSetSampleMaskEXT cmdSetSampleMaskEXT = nullptr;
    inline PFN_vkCmdSetAlphaToCoverageEnableEXT cmdSetAlphaToCoverageEnableEXT = nullptr;
    inline PFN_vkCmdBindShadersEXT cmdBindShadersEXT = nullptr;

    inline PFN_vkGetDescriptorSetLayoutBindingOffsetEXT getDescriptorSetLayoutBindingOffsetEXT = nullptr;
    inline PFN_vkGetDescriptorSetLayoutSizeEXT getDescriptorSetLayoutSizeEXT = nullptr;
    inline PFN_vkGetDescriptorEXT getDescriptorEXT = nullptr;

    inline PFN_vkCmdBindDescriptorBuffersEXT cmdBindDescriptorBuffersEXT = nullptr;
    inline PFN_vkCmdSetDescriptorBufferOffsetsEXT cmdSetDescriptorBufferOffsetsEXT = nullptr;

    inline PFN_vkCmdPushDescriptorSetKHR cmdPushDescriptorSetKHR = nullptr;

    // dynamic state that shader objects need , and that has no core equivalent
    inline PFN_vkCmdSetColorBlendEnableEXT cmdSetColorBlendEnableEXT = nullptr;
    inline PFN_vkCmdSetColorWriteMaskEXT cmdSetColorWriteMaskEXT = nullptr;
    inline PFN_vkCmdSetColorBlendEquationEXT cmdSetColorBlendEquationEXT = nullptr;
    inline PFN_vkCmdSetFragmentShadingRateKHR cmdSetFragmentShadingRateKHR = nullptr;

    inline PFN_vkGetShaderBinaryDataEXT getShaderBinaryDataEXT = nullptr;
    inline PFN_vkDestroyShaderEXT destroyShaderEXT = nullptr;

    inline PFN_vkCmdBeginConditionalRenderingEXT cmdBeginConditionalRenderingEXT = nullptr;
    inline PFN_vkCmdEndConditionalRenderingEXT cmdEndConditionalRenderingEXT = nullptr;



    inline const std::string getFunctionName(const std::string PFN_function){
        if(PFN_function.size() < 4) return {};
        return {PFN_function.substr(4)};
    }

    template<typename Function>
    void loadFunction(VkInstance instance,  const std::string name,  Function &function){
        function = (Function) vkGetInstanceProcAddr(instance , name.c_str());
    }


    inline VkTransformMatrixKHR matrixTransform34(const glm::mat4& m) {
        VkTransformMatrixKHR out;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 4; col++) {
                out.matrix[row][col] = m[col][row];
            }
        }
        return out;
    }


    inline VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity , VkDebugUtilsMessageTypeFlagsEXT flag  , const VkDebugUtilsMessengerCallbackDataEXT* pMessage, void* user){
        std::cout << pMessage->pMessage << '\n';
        return VK_FALSE;
    }


    inline VkResult createDebugUtilsMessengerEXT(VkInstance instance , VkDebugUtilsMessengerCreateInfoEXT* info , const VkAllocationCallbacks* alloc , VkDebugUtilsMessengerEXT* debug){
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance , "vkCreateDebugUtilsMessengerEXT");
        if(!func) return VK_ERROR_EXTENSION_NOT_PRESENT;
        else{
            return func(instance , info , alloc , debug);
        }
    }


    inline void destroyDebugUtilsMessengerEXT(VkInstance instance , const VkAllocationCallbacks* alloc , VkDebugUtilsMessengerEXT debug){
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance , "vkDestroyDebugUtilsMessengerEXT");
        if(func) func(instance , debug , alloc);
    }

    class Instance final{
    private:
        void clean(){
            if(enableValidations){
                if(debug) destroyDebugUtilsMessengerEXT(instance , nullptr , debug);
                EngineLogger_T::messageCreateInfo info{};
                info.errorType = ErrorSignature::ErrorTypeNONE;
                info.message = std::format("the debugger for instance , of typical instance has been deleted");
                engineLogger(&info);
            }   

            if(instance){
                vkDestroyInstance(instance , nullptr);
                EngineLogger_T::messageCreateInfo info{};
                info.errorType = ErrorSignature::ErrorTypeNONE;
                info.message = std::format("the instance  of typical instance has been deleted");
                engineLogger(&info);
            }

        }

        inline void setupDebugUtils(VkDebugUtilsMessengerCreateInfoEXT &info){
            info = {};
            info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            info.pfnUserCallback = debugUtilsMessengerCallback;
            info.pUserData = nullptr;
            info.flags = 0;
            info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        }
            
        
        [[nodiscard]] bool checkValidations(){
            if(!enableValidations) return true;
            uint32_t count;
            vkEnumerateInstanceLayerProperties(&count , nullptr);
            std::vector<VkLayerProperties> layers(count);
            vkEnumerateInstanceLayerProperties(&count , layers.data());
            
            for(auto& readLayer : validationLayers){
                bool founded = false;
                for(auto& layer : layers){
                    if(strcmp(readLayer , layer.layerName) == 0 ){
                        founded = true;
                        break;
                    }
                }
                if(!founded) return false;

            }

            return true;
        }

        std::vector<const char*> getExtensions(){
            uint32_t count;
            auto data = glfwGetRequiredInstanceExtensions(&count);
            std::vector<const char*> ext(data , data + count);
            if(enableValidations) ext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            return ext;
        }

        ErrorDataOutput<void> createInstance(){
            if(enableValidations && !checkValidations()) throw std::runtime_error("CRITICAL ERROR: validation layers are not supported");
            VkApplicationInfo app{};
            app.applicationVersion = VK_MAKE_VERSION(1 , 0 , 0);
            app.apiVersion = VK_API_VERSION_1_3;
            app.engineVersion = VK_MAKE_VERSION(1 , 0 , 0);
            app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app.pNext = nullptr;
            app.pEngineName = "null";
            app.pApplicationName = "null";

            auto ext = getExtensions();
            VkDebugUtilsMessengerCreateInfoEXT dInfo{};
            VkInstanceCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            info.pApplicationInfo = &app;
            info.enabledExtensionCount = static_cast<uint32_t>(ext.size());
            info.ppEnabledExtensionNames = ext.data();
            info.flags = 0;
            if(enableValidations){
                setupDebugUtils(dInfo);
                info.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                info.ppEnabledLayerNames = validationLayers.data();
                info.pNext = &dInfo;
            }else{
                info.enabledLayerCount = 0;
                info.ppEnabledLayerNames = nullptr;
                info.pNext = nullptr;
            }

            if(vkCreateInstance(&info , nullptr , &instance) != VK_SUCCESS){
                ErrorDataType errorMessage = ErrorDataType(ErrorSignature::ErrorTypeCRITICAL , "failed to create instance, error is critical , project resets now");
                return std::unexpected<ErrorDataType>(errorMessage);
            }

            if(enableValidations){
                if(createDebugUtilsMessengerEXT(instance , &dInfo , nullptr , &debug) != VK_SUCCESS){
                    ErrorDataType errorMessage = ErrorDataType(ErrorSignature::ErrorTypeCRITICAL , "failed to create debug, error is critical , project resets now");
                    return std::unexpected<ErrorDataType>(errorMessage);
                }
            }

            return {};
        }
        

    public:

        Instance(Instance& input) = delete;
        Instance(Instance&& input){
            instance = std::exchange(input.instance , nullptr);
            debug = std::exchange(input.debug , nullptr);

        }

        Instance& operator=(Instance& input) = delete;
        Instance& operator=(Instance&& input){
            instance = std::exchange(input.instance , nullptr);
            debug = std::exchange(input.debug , nullptr);
            return *this;
        }

        Instance(){
            auto errorAnalys = createInstance();
            if(errorAnalys.has_value()){
                EngineLogger_T::messageCreateInfo info{};
                info.errorType = ErrorSignature::ErrorTypeSUCCESS;
                info.message = "instance has been created successfully\n";
                engineLogger(&info);
            }else{
                engineLogger(&errorAnalys.error());
                throw std::runtime_error("error occured , check log");
            }
        }

        VkInstance instance = nullptr;
        VkDebugUtilsMessengerEXT debug = nullptr;


        ~Instance(){
            clean();
        }
    };
}