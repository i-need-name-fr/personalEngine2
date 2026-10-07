#pragma once

#include "PhysicalDevice.hpp"

namespace mox{

    class LogicalDevice final{
    private:
        
        [[nodiscard]] ErrorDataOutput<void> createLogicalDevice(VkPhysicalDevice pDevice , VkSurfaceKHR surface, const uint32_t queueCount){
            std::vector<float> priority(queueCount , 1.0f);
            DetailsCreateInfo det{};
            det.device = pDevice;
            det.surface = surface;
            auto details = queueFamilyDetails::queryQueueFamilyDetails(det);
            std::set<uint32_t> indexes = {details.graphics.value() , details.present.value()};
            std::vector<VkDeviceQueueCreateInfo> queues;
            for(auto& index : indexes){
                VkDeviceQueueCreateInfo info{};
                info.queueFamilyIndex = index;
                info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                info.pQueuePriorities = priority.data();
                info.queueCount = queueCount;
                info.flags = 0;
                info.pNext = nullptr;
                queues.push_back(info);
            }

            // vkCmdDrawIndexedIndirectCount needs this , and it only exists on the
            // Vulkan 1.2 feature struct
            // One struct , not four. VkPhysicalDeviceVulkan12Features subsumes descriptor
            // indexing , timeline semaphores and buffer device address , and the spec
            // forbids mixing it with the individual structs it replaces.
            VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional{};
            conditional.conditionalRendering = VK_TRUE;
            conditional.pNext = nullptr;
            conditional.sType= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT;

            VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR images{};
            images.pNext = &conditional;
            images.unifiedImageLayouts = VK_TRUE;
            images.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR;

            VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRate{};
            shadingRate.primitiveFragmentShadingRate = VK_TRUE;
            shadingRate.attachmentFragmentShadingRate = VK_TRUE;
            shadingRate.pipelineFragmentShadingRate = VK_TRUE;
            shadingRate.pNext = &images;
            shadingRate.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;

            VkPhysicalDeviceSynchronization2Features sync2Features{};
            sync2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
            sync2Features.pNext = &shadingRate;
            sync2Features.synchronization2 = VK_TRUE;

            VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBuffer{};
            descriptorBuffer.descriptorBuffer = VK_TRUE;
            descriptorBuffer.descriptorBufferCaptureReplay = VK_TRUE;
            descriptorBuffer.pNext = &sync2Features;
            descriptorBuffer.descriptorBufferPushDescriptors = VK_TRUE;
            descriptorBuffer.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;

            VkPhysicalDeviceVulkan12Features vulkan12{};
            vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

            // vkCmdDrawIndexedIndirectCount
            vulkan12.drawIndirectCount = VK_TRUE;

            vulkan12.timelineSemaphore = VK_TRUE;
            vulkan12.bufferDeviceAddress = VK_TRUE;

            // the extension is enabled , so this has to be on as well
            vulkan12.descriptorIndexing = VK_TRUE;
            // Texture2D tex[] in the shader - an array with no compile time size
            vulkan12.runtimeDescriptorArray = VK_TRUE;
            // indexing that array with a value that differs between lanes
            vulkan12.shaderSampledImageArrayNonUniformIndexing  = VK_TRUE;
            vulkan12.shaderStorageImageArrayNonUniformIndexing  = VK_TRUE;
            vulkan12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
            vulkan12.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
            // slots may stay unwritten as long as the shader does not read them
            vulkan12.descriptorBindingPartiallyBound = VK_TRUE;
            // the real length is given at allocation time
            vulkan12.descriptorBindingVariableDescriptorCount = VK_TRUE;
            // rewrite a descriptor while the set is still bound
            vulkan12.descriptorBindingSampledImageUpdateAfterBind  = VK_TRUE;
            vulkan12.descriptorBindingStorageImageUpdateAfterBind  = VK_TRUE;
            vulkan12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
            vulkan12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
            vulkan12.descriptorBindingUpdateUnusedWhilePending     = VK_TRUE;
            vulkan12.pNext = &descriptorBuffer;

            VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
            shaderObjectFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
            shaderObjectFeatures.shaderObject = VK_TRUE;
            shaderObjectFeatures.pNext = &vulkan12;

            VkPhysicalDeviceDynamicRenderingFeatures dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
            dynamic.dynamicRendering = VK_TRUE;
            dynamic.pNext = &shaderObjectFeatures;

            VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
            asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            asFeatures.accelerationStructure = VK_TRUE;
            asFeatures.pNext = &dynamic;

            VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeature{};
            rayTracingFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
            rayTracingFeature.pNext = &asFeatures;
            rayTracingFeature.rayTracingPipeline = VK_TRUE;

            VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeature{};
            rayQueryFeature.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            rayQueryFeature.rayQuery = VK_TRUE;
            rayQueryFeature.pNext    = &rayTracingFeature;

            VkPhysicalDeviceFeatures features{};
            features.samplerAnisotropy = VK_TRUE;
            // gFragment.hlsl writes the G buffer through storage images and storage
            // buffers , which the fragment stage may not do without this
            features.fragmentStoresAndAtomics = VK_TRUE;
            // the cull shader writes a non zero firstInstance into every indirect
            // draw , and that is only legal with this feature on. The value lives in
            // a GPU buffer , so the validation layer never sees it - without this the
            // draws quietly read instance 0 and the whole grid stacked into one cube
            features.drawIndirectFirstInstance = VK_TRUE;
            // the cull shader fills one draw per mesh into the indirect buffer ( drawCount > 1 )
            features.multiDrawIndirect = VK_TRUE;

            VkDeviceCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            info.flags = 0;
            info.pEnabledFeatures= &features;
            info.pNext = &rayQueryFeature;
            info.enabledExtensionCount = static_cast<uint32_t>(requiredExt.size());
            info.ppEnabledExtensionNames = requiredExt.data();
            info.pQueueCreateInfos = queues.data();
            info.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
            info.enabledLayerCount = (enableValidations ? static_cast<uint32_t>(validationLayers.size()) : 0);
            info.ppEnabledLayerNames = (enableValidations ? validationLayers.data() : nullptr);

            const auto result = vkCreateDevice(pDevice , &info , nullptr , &device);

            if(result != VK_SUCCESS){
                ErrorDataType error{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt create the logical device , error ID -> {}" , static_cast<int32_t>(result))};
                return std::unexpected(error);
            }
            return {};
        }

    public: 

        LogicalDevice() = delete;

        struct logicalDeviceCreateInfo {
            VkPhysicalDevice device;
            VkSurfaceKHR surface;
            uint32_t requiredForGraphics;
            uint32_t requiredForMultiThread;
        };

        LogicalDevice(const logicalDeviceCreateInfo& info){
            if(!info.device || !info.surface) throw std::runtime_error("couldnt create the logicalDevice , since the info is nullptr");
            if(info.requiredForGraphics + info.requiredForMultiThread > 16) throw std::runtime_error("couldnt create the logical device since the required queues is more than 16");
            auto result = createLogicalDevice(info.device , info.surface,  info.requiredForGraphics + info.requiredForMultiThread);
            if(result.has_value()){
                ErrorDataType res = {MOX_ERROR_TYPE_SUCCESS , "logical device has been created "};
                engineLogger(&res);
            }else{
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }


        LogicalDevice(LogicalDevice& input) = delete;
        LogicalDevice(LogicalDevice&& input){
            device = std::exchange(input.device , nullptr);
        }

        LogicalDevice& operator=(LogicalDevice& input) = delete;
        LogicalDevice& operator=(LogicalDevice&& input){
            device = std::exchange(input.device , nullptr);
            return *this;
        }

        VkDevice device = nullptr;

        ~LogicalDevice(){
            if(device) vkDestroyDevice(device , nullptr);
            ErrorDataType res{MOX_ERROR_TYPE_NONE , "deleting the logical device"};
            engineLogger(&res);
        }
    };
}