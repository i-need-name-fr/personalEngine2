#pragma once

#include "LogicalDevice.hpp"
#include <vma/vk_mem_alloc.h>

namespace mox{

    class MemoryAllocator final{
    public:

        SetTypicalOperators(MemoryAllocator)
        SetTypicalTransferOperators(MemoryAllocator)

        struct CreateInfo{
            VkInstance instance{};
            VkDevice logicalDevice{};
            VkPhysicalDevice pDevice{};
        };

        MemoryAllocator(const CreateInfo &info){
            if(!info.instance || !info.logicalDevice || !info.pDevice){
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create memory allocation , since the info contains null data"});
                THROW_MESSAGE;
            }

            const auto result = createAllocater(info);
            if(!result.has_value()){
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        VmaAllocator allocation{nullptr};

        ~MemoryAllocator(){
            vmaDestroyAllocator(allocation);
        }

    private:
        
        [[nodiscard]] ErrorDataOutput<void> createAllocater(const CreateInfo &input)noexcept{

            VmaVulkanFunctions functions{};

            functions.vkCreateBuffer              = vkCreateBuffer;
            functions.vkBindBufferMemory          = vkBindBufferMemory;
            functions.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
            functions.vkBindImageMemory           = vkBindImageMemory;
            functions.vkAllocateMemory            = vkAllocateMemory;
            functions.vkFreeMemory                = vkFreeMemory;
            functions.vkMapMemory                 = vkMapMemory;
            functions.vkUnmapMemory               = vkUnmapMemory;
            functions.vkCreateImage               = vkCreateImage;
            functions.vkGetInstanceProcAddr       = vkGetInstanceProcAddr;
            functions.vkGetDeviceProcAddr         = vkGetDeviceProcAddr;
            functions.vkGetPhysicalDeviceProperties       = vkGetPhysicalDeviceProperties;
            functions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
            functions.vkFlushMappedMemoryRanges           = vkFlushMappedMemoryRanges;
            functions.vkInvalidateMappedMemoryRanges      = vkInvalidateMappedMemoryRanges;
            functions.vkGetImageMemoryRequirements        = vkGetImageMemoryRequirements;
            functions.vkDestroyBuffer                     = vkDestroyBuffer;
            functions.vkDestroyImage                      = vkDestroyImage;
            functions.vkCmdCopyBuffer                     = vkCmdCopyBuffer;
            functions.vkGetBufferMemoryRequirements2KHR       = vkGetBufferMemoryRequirements2;
            functions.vkGetImageMemoryRequirements2KHR        = vkGetImageMemoryRequirements2;
            functions.vkBindBufferMemory2KHR                  = vkBindBufferMemory2;
            functions.vkBindImageMemory2KHR                   = vkBindImageMemory2;
            functions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
            functions.vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements;
            functions.vkGetDeviceImageMemoryRequirements  = vkGetDeviceImageMemoryRequirements;

            VmaAllocatorCreateInfo info{};
            info.vulkanApiVersion = VK_API_VERSION_1_4;
            info.physicalDevice   = input.pDevice;
            info.device           = input.logicalDevice;
            info.instance         = input.instance;
            info.pVulkanFunctions  = &functions;
            info.flags =VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT
                     | VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

            const auto result = vmaCreateAllocator(&info , &allocation);
            if(result != VK_SUCCESS) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create vma allocator ,error ID -> {}", static_cast<int32_t>(result))});
            return {};
        }

        void processCopy(MemoryAllocator&& input) noexcept {
            processDelete();

            allocation = std::exchange(input.allocation , nullptr);
        }

        void processDelete() noexcept{
            if(allocation) vmaDestroyAllocator(allocation);
            allocation = nullptr;
        }
    };
}