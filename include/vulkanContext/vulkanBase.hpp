#ifndef base_HPP
#define base_HPP

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../shaderStructures/shaderStructure.h"

#include <iostream>
#include "../logger/logger.hpp"

#include <memory>

#include "Instance.hpp"
#include "FamilyDetails.hpp"
#include "Window.hpp"
#include "Surface.hpp"
#include "Player.hpp"
#include "PhysicalDevice.hpp"
#include "LogicalDevice.hpp"
#include "SwapChain.hpp"
#include "Queue.hpp"
#include "GPUMultiThread.hpp"

namespace mox{

    using pPlayer = std::shared_ptr<Player>;
    using pWindow = std::shared_ptr<Window>;
    using pInstance = std::shared_ptr<Instance>;
    using pSurface = std::shared_ptr<Surface>;
    using pPhysicalDevice = std::shared_ptr<PhysicalDevice>;
    using pLogicalDevice = std::shared_ptr<LogicalDevice>;
    using pSwapchain = std::shared_ptr<Swapchain>;


    class VulkanContext final{
    private:

        void init(const uint32_t m , const uint32_t am){
            try{
                user = std::make_shared<Player>();

                mox::Window::WindowCreateInfo windowInfo{};

                windowInfo.pUser = user.get();
                windowInfo.key = Player::keyCallBack;
                windowInfo.mouse = Player::mousePosCallBack;

                window = std::make_shared<Window>(windowInfo);

                instance = std::make_shared<Instance>();

                Surface::SurfaceCreateInfo surfaceInfo{};
                surfaceInfo.instance = instance->instance;
                surfaceInfo.window = window->window;

                surface = std::make_shared<Surface>(surfaceInfo);

                PhysicalDevice::PhysicalDeviceCreateInfo pDeviceInfo{};
                pDeviceInfo.instance = instance->instance;
                pDeviceInfo.surface = surface->surface;

                physicalDevice = std::make_shared<PhysicalDevice>(pDeviceInfo);

                LogicalDevice::logicalDeviceCreateInfo logicalInfo{};
                logicalInfo.device = physicalDevice->device;
                logicalInfo.surface = surface->surface;

                logicalInfo.requiredForGraphics = 2;
                logicalInfo.requiredForMultiThread = m + am;

                logicalDevice = std::make_shared<LogicalDevice>(logicalInfo);

                // now the init for queues
                {
                    Queue::CreateInfo info{};
                    info.countIndex = 0;
                    info.device = logicalDevice->device;
                    info.pDevice = physicalDevice->device;
                    info.needsCmd = true;
                    info.surface = surface->surface;

                    graphicsQueue = std::make_shared<Queue>(info);

                    info.countIndex = 1;

                    VirtualShadowQueue = std::make_shared<QueueLane>(info);
                }

                {
                    // we used only 0 and 1 , so offset is 2
                    QueueManager::CreateInfo info{};
                    info.device = logicalDevice->device;
                    info.pDevice = physicalDevice->device;
                    info.firstIndex = 2;
                    info.threadsCount = m;
                    info.surface = surface->surface;

                    queueManager = std::make_shared<QueueManager>(info);
                }

                // later for queue multithread
                {
                    const uint32_t offset = m + 2;
                }

                Swapchain::SwapchainCreateInfo info{};
                info.device = logicalDevice->device;
                info.pDevice = physicalDevice->device;
                info.surface = surface->surface;
                info.window = window->window;

                sc = std::make_shared<Swapchain>(info);

                loadAllFunctions(instance->instance);
            }
            catch(std::exception& e){throw;}
        }

        inline void loadAllFunctions(VkInstance instance){
            loadFunction(instance, getFunctionName("PFN_vkCreateAccelerationStructureKHR"), createAccelerationStructureKHR);
            loadFunction(instance, getFunctionName("PFN_vkGetAccelerationStructureBuildSizesKHR"), getAccelerationStructureBuildSizesKHR);
            loadFunction(instance, getFunctionName("PFN_vkCmdBuildAccelerationStructuresKHR"), cmdBuildAccelerationStructuresKHR);
            loadFunction(instance, getFunctionName("PFN_vkDestroyAccelerationStructureKHR"), destroyAccelerationStructureKHR);
            loadFunction(instance, getFunctionName("PFN_vkGetAccelerationStructureDeviceAddressKHR"), getAccelerationStructureDeviceAddressKHR);

            loadFunction(instance, getFunctionName("PFN_vkCreateRayTracingPipelinesKHR"), createRayTracingPipelinesKHR);
            loadFunction(instance, getFunctionName("PFN_vkGetRayTracingShaderGroupHandlesKHR"), getRayTracingShaderGroupHandlesKHR);
            loadFunction(instance, getFunctionName("PFN_vkCmdTraceRaysKHR"), cmdTraceRaysKHR);

            loadFunction(instance, getFunctionName("PFN_vkCreateShadersEXT"), createShaderEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetVertexInputEXT"), cmdSetVertexInputEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetPolygonModeEXT"), cmdSetPolygonModeEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetRasterizationSamplesEXT"), cmdSetRasterizationSamplesEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetSampleMaskEXT"), cmdSetSampleMaskEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetAlphaToCoverageEnableEXT"), cmdSetAlphaToCoverageEnableEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdBindShadersEXT"), cmdBindShadersEXT);

            loadFunction(instance, getFunctionName("PFN_vkGetShaderBinaryDataEXT"), getShaderBinaryDataEXT);
            loadFunction(instance, getFunctionName("PFN_vkDestroyShaderEXT"), destroyShaderEXT);

            loadFunction(instance, getFunctionName("PFN_vkCmdPushDescriptorSetKHR"), cmdPushDescriptorSetKHR);

            loadFunction(instance, getFunctionName("PFN_vkCmdSetColorBlendEnableEXT"), cmdSetColorBlendEnableEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetColorWriteMaskEXT"), cmdSetColorWriteMaskEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetColorBlendEquationEXT"), cmdSetColorBlendEquationEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetFragmentShadingRateKHR"), cmdSetFragmentShadingRateKHR);

            loadFunction(instance, getFunctionName("PFN_vkGetDescriptorSetLayoutBindingOffsetEXT"), getDescriptorSetLayoutBindingOffsetEXT);
            loadFunction(instance, getFunctionName("PFN_vkGetDescriptorSetLayoutSizeEXT"), getDescriptorSetLayoutSizeEXT);
            loadFunction(instance, getFunctionName("PFN_vkGetDescriptorEXT"), getDescriptorEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdBindDescriptorBuffersEXT"), cmdBindDescriptorBuffersEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdSetDescriptorBufferOffsetsEXT"), cmdSetDescriptorBufferOffsetsEXT);

            loadFunction(instance, getFunctionName("PFN_vkCmdBeginConditionalRenderingEXT"), cmdBeginConditionalRenderingEXT);
            loadFunction(instance, getFunctionName("PFN_vkCmdEndConditionalRenderingEXT"), cmdEndConditionalRenderingEXT);

        }

    public:

        struct CreateInfo{
            uint32_t asyncMultiThreadQueueCount;
            uint32_t multithreadQueueCount;
        };

        explicit VulkanContext(const CreateInfo &info){
            init(info.multithreadQueueCount , info.asyncMultiThreadQueueCount);
        }

        VulkanContext(const VulkanContext&) = delete;
        VulkanContext& operator=(const VulkanContext&) = delete;
        VulkanContext(VulkanContext&& input) noexcept = default;
        VulkanContext& operator=(VulkanContext&& input) noexcept = default;
        ~VulkanContext() = default;
        
        pPlayer user = nullptr;
        pWindow window = nullptr;
        pInstance instance = nullptr;
        pSurface surface = nullptr;

        pPhysicalDevice physicalDevice = nullptr;
        pLogicalDevice logicalDevice = nullptr;

        pSwapchain sc = nullptr;

        pQueueManager queueManager = nullptr;
        pQueue graphicsQueue = nullptr;
        pQueueLane VirtualShadowQueue = nullptr;
        
    };

    inline uint32_t findMemoryIndex(VulkanContext* base , uint32_t index , VkMemoryPropertyFlags flags){
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(base->physicalDevice->device , &properties);
        for(uint8_t i = 0 ; i < properties.memoryTypeCount ;  i++){
            if ( index & ( 1 << i) && (properties.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }

        throw std::runtime_error("couldnt find the memory index for these flags");
    }

    using pVulkanContext = std::shared_ptr<VulkanContext>;

    inline uint32_t alignUp(uint32_t size , uint32_t alignment) noexcept {
        return (alignment == 0) ? size : ((size + alignment - 1) / alignment) * alignment;
    }

    enum class VulkanRequirements : uint32_t{
        Instance = 0x01,
        Queues = 0x02,
        QueueManager = 0x03,
        LogicalDevice = 0x04,
        PhysicalDevice = 0x05,
        Swapchain = 0x06
    };

    inline static constexpr uint32_t VkReqTypeInstance = static_cast<uint32_t>(VulkanRequirements::Instance);
    inline static constexpr uint32_t VkReqTypeQueues = static_cast<uint32_t>(VulkanRequirements::Queues);
    inline static constexpr uint32_t VkReqTypeQueueManager = static_cast<uint32_t>(VulkanRequirements::QueueManager);
    inline static constexpr uint32_t VkReqTypeLogicalDevice = static_cast<uint32_t>(VulkanRequirements::LogicalDevice);
    inline static constexpr uint32_t VkReqTypePhysicalDevice = static_cast<uint32_t>(VulkanRequirements::PhysicalDevice);
    inline static constexpr uint32_t VkReqTypeSwapchain = static_cast<uint32_t>(VulkanRequirements::Swapchain);


    [[nodiscard]] [[gnu::hot]] inline ErrorDataOutput<void> checkRequirements(const uint32_t reqs , VulkanContext* context) noexcept{

        if(!context){
            return  std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "the requirement for context is not valid, the context is nullptr"});
        }

        if((reqs & VkReqTypeInstance) == VkReqTypeInstance){
            if(!context->instance) return  std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "the requirement for context is not valid, the instance is nullptr"});
        }

        if((reqs & VkReqTypeQueues) == VkReqTypeQueues){
            if(!context->graphicsQueue) return  std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "the requirement for context is not valid, the queues is nullptr"});
        }

        if((reqs &VkReqTypeQueueManager) ==VkReqTypeQueueManager){
            if(!context->queueManager) return  std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "the requirement for context is not valid, the queue manager is nullptr"});
        }

        if((reqs & VkReqTypeLogicalDevice) == VkReqTypeLogicalDevice){
            if(!context->logicalDevice) return  std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "the requirement for context is not valid, the logical device is nullptr"});
        }

        if((reqs & VkReqTypePhysicalDevice) == VkReqTypePhysicalDevice){
            if(!context->physicalDevice) return  std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "the requirement for context is not valid, the physical device is nullptr"});
        }

        if((reqs & VkReqTypeSwapchain) == VkReqTypeSwapchain){
            if(!context->sc) return  std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "the requirement for context is not valid, the swapchain is nullptr"});
        }

        return {};
    }
}

#endif