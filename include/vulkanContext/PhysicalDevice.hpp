#pragma once

#include "Instance.hpp"
#include "FamilyDetails.hpp"
#include <set>

namespace mox{



    class PhysicalDevice{
    private:

        VkInstance instance = nullptr;

        // check for all suitable extensions
        // returns true if the all extension are included
        [[nodiscard]] bool checkExtensions(VkPhysicalDevice device){
            uint32_t count;
            vkEnumerateDeviceExtensionProperties(device , nullptr , &count , nullptr);
            std::vector<VkExtensionProperties> exts(count);
            vkEnumerateDeviceExtensionProperties(device , nullptr , &count , exts.data());
            std::set<std::string> required(requiredExt.begin() , requiredExt.end());
            for(auto& e : exts) required.erase(e.extensionName);
            return required.empty();
        }

        // since we got all the required functions , we can make the check 

        bool suitableDevice(VkPhysicalDevice device , VkSurfaceKHR surface){
            if(!device) return false;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(device , &properties);
            if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){
                bool suitableExt = checkExtensions(device);
                bool suitableSC = false;
                DetailsCreateInfo info{};
                info.device = device;
                info.surface = surface;
                bool suitableQueue = queueFamilyDetails::queryQueueFamilyDetails(info).complete();
                if(suitableExt){
                    suitableSC = swapchainDetails::querySwapchainDetails(info).complete();
                }
                return suitableExt && suitableQueue && suitableSC;
            }
            return false;
        }

        [[nodiscard]] ErrorDataOutput<void> pickSuitableDevice(VkSurfaceKHR surface){
            uint32_t count;
            vkEnumeratePhysicalDevices(instance , &count , nullptr);
            std::vector<VkPhysicalDevice> devices(count);
            vkEnumeratePhysicalDevices(instance , &count , devices.data());
            this->device = nullptr;
            for(auto& device : devices){
                if(suitableDevice(device , surface)) this->device = device;
            }
            if(!this->device){
                ErrorDataType error = {MOX_ERROR_TYPE_CRITICAL , "couldnt find the suitable GPU"};
                return std::unexpected(error);
            }
            return {};
        }

    public:
        
        struct PhysicalDeviceCreateInfo{
            VkInstance instance;
            VkSurfaceKHR surface;
        };

        PhysicalDevice() = delete;
        PhysicalDevice(const PhysicalDeviceCreateInfo& info){
            if(!info.instance || !info.surface) throw std::runtime_error("couldnt find the suitable GPU , since the info has nullptr");
            this->instance = info.instance;
            auto result = pickSuitableDevice(info.surface);
            if(result.has_value()){
                ErrorDataType res = {MOX_ERROR_TYPE_SUCCESS , "founded the suitable GPU"};
                engineLogger(&res);
            }else{
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        PhysicalDevice(PhysicalDevice& input) = delete;
        PhysicalDevice(PhysicalDevice&& input){
            device = std::exchange(input.device , nullptr);
            instance = std::exchange(input.instance , nullptr);
        }

        PhysicalDevice& operator=(PhysicalDevice& input) = delete;
        PhysicalDevice& operator=(PhysicalDevice&& input){
            device = std::exchange(input.device , nullptr);
            instance = std::exchange(input.instance , nullptr);
            return *this;
        }

        VkPhysicalDevice device = nullptr;


        ~PhysicalDevice(){
            // none since we got nothing to destroy , lol 
            ErrorDataType res{MOX_ERROR_TYPE_NONE , "deleting the physical device"};
            engineLogger(&res);
        }
    };
}