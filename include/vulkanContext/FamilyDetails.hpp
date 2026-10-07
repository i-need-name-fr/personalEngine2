#pragma once

#include "Instance.hpp"

namespace mox{

    struct DetailsCreateInfo{
            VkPhysicalDevice device;
            VkSurfaceKHR surface;
    };

    struct queueFamilyDetails{
        std::optional<uint32_t> graphics;
        std::optional<uint32_t> present;

        [[nodiscard]] bool complete(){
            return graphics.has_value() && present.has_value();
        }



        [[gnu::cold]] static queueFamilyDetails queryQueueFamilyDetails(const DetailsCreateInfo& info){
            queueFamilyDetails details{};

            [[unlikely]] if(!info.device || !info.surface)  return {};

            uint32_t count;
            vkGetPhysicalDeviceQueueFamilyProperties(info.device , &count , nullptr);
            std::vector<VkQueueFamilyProperties> queues(count);
            vkGetPhysicalDeviceQueueFamilyProperties(info.device , &count , queues.data());

            uint32_t i = 0;
            for(auto& queue : queues){
                if(queue.queueFlags & VK_QUEUE_GRAPHICS_BIT && queue.queueCount > 1) details.graphics = i;

                VkBool32 supported;
                vkGetPhysicalDeviceSurfaceSupportKHR(info.device , i , info.surface , &supported);
                if(supported) details.present = i;
                if(details.complete()) break;
                i++;
            }   

            return details;
        }
    };

    struct swapchainDetails{
        VkSurfaceCapabilitiesKHR cap;
        std::vector<VkSurfaceFormatKHR> formats{};
        std::vector<VkPresentModeKHR> modes{};

        [[nodiscard]] bool complete(){
            return !modes.empty() && !formats.empty();
        }
        
        [[gnu::cold]] static swapchainDetails querySwapchainDetails(const DetailsCreateInfo& info){
            swapchainDetails details{};
            [[unlikely]] if(!info.device || !info.surface)  return {};
            
            // capabilities
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(info.device , info.surface , &details.cap);
            // formats
            {
                uint32_t count;
                vkGetPhysicalDeviceSurfaceFormatsKHR(info.device , info.surface , &count , nullptr);
                if(count){
                    details.formats.resize(count);
                    vkGetPhysicalDeviceSurfaceFormatsKHR(info.device , info.surface , &count , details.formats.data());
                }
            }

            {
                uint32_t count;
                vkGetPhysicalDeviceSurfacePresentModesKHR(info.device , info.surface , &count, nullptr);
                if(count){
                    details.modes.resize(count);
                    vkGetPhysicalDeviceSurfacePresentModesKHR(info.device , info.surface , &count, details.modes.data());
                }
            }
            return details;
        }

    };
}