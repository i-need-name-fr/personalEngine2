#pragma once 
#include "LogicalDevice.hpp"

namespace mox{

    class Swapchain final{
    private:

        VkDevice device = nullptr;
        

        [[gnu::cold]] inline void chooseFormat(std::vector<VkSurfaceFormatKHR> const &format) {
            if(format.empty()) throw std::runtime_error("the surface reports no formats at all");

            // The SVGF resolve writes the swapchain image as a storage image and
            // its HLSL declares [[vk::image_format("rgba8")]]. A storage image
            // format has to match the shader EXACTLY, so R8G8B8A8_UNORM is not a
            // preference here, it is a requirement - B8G8R8A8_UNORM would make
            // every store undefined. _SRGB cannot be a storage image at all.
            for(auto& f : format){
                if(f.format == VK_FORMAT_R8G8B8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
                    this->format = f;
                    return;
                }
            }

            for(auto& f : format){
                if(f.format == VK_FORMAT_R8G8B8A8_UNORM){
                    this->format = f;
                    return;
                }
            }

            // without a fallback `format` simply stayed uninitialised here
            throw std::runtime_error("the surface does not expose VK_FORMAT_R8G8B8A8_UNORM, which the SVGF resolve writes into as a storage image");
        }

        [[gnu::cold]] inline VkPresentModeKHR chooseSwapchainPresentMode(const std::vector<VkPresentModeKHR> &modes) noexcept {
            for(auto& mode : modes){
                if(mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
            }
            std::cerr << "couldnt find the suitable present mode for swapchain , but since its not really a critical error, we will take the first one\n";
            return modes[0];
        }

        
        void chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR cap , GLFWwindow* window){
            if(std::numeric_limits<uint32_t>::max() != cap.currentExtent.width){
                extent = cap.currentExtent;
            }else{
                VkExtent2D extent;
                int width_temp , height_temp;
                uint32_t width , height;
                glfwGetFramebufferSize(window , &width_temp , &height_temp);
                width = static_cast<uint32_t>(width_temp);
                height = static_cast<uint32_t>(height_temp);
                extent.width = std::clamp(width , cap.minImageExtent.width , cap.maxImageExtent.width);
                extent.height = std::clamp(height , cap.minImageExtent.height , cap.maxImageExtent.height);
                this->extent = extent;
            }
        }

        [[nodiscard]] ErrorDataOutput<void> init(VkSurfaceKHR surface,  GLFWwindow* window , VkPhysicalDevice pDevice) noexcept {
            DetailsCreateInfo det{};
            det.device = pDevice;
            det.surface = surface;
            auto details = swapchainDetails::querySwapchainDetails(det);
            auto mode = chooseSwapchainPresentMode(details.modes);
            chooseSwapchainExtent(details.cap , window);
            chooseFormat(details.formats);

            uint32_t imageCount = details.cap.minImageCount +1;
            if(imageCount > details.cap.maxImageCount && details.cap.maxImageCount > 0) imageCount = details.cap.maxImageCount;

            VkSwapchainCreateInfoKHR info{};
            info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            info.flags = 0;
            info.pNext = nullptr;
            info.imageExtent = extent;
            info.imageFormat = format.format;
            info.imageColorSpace = format.colorSpace;
            info.presentMode = mode;
            info.preTransform = details.cap.currentTransform;
            info.clipped = VK_TRUE;
            info.oldSwapchain = nullptr;
            info.imageArrayLayers = 1;
            info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT ;
            info.surface = surface;
            info.minImageCount = imageCount;
            
            if(vkCreateSwapchainKHR(device , &info ,nullptr , &sc) != VK_SUCCESS){
                ErrorDataType error = {MOX_ERROR_TYPE_FAILED_CREATION  ,"couldnt create the swapchain"};
                return std::unexpected(error);
            }
            vkGetSwapchainImagesKHR(device , sc , &this->imageCount , nullptr);
            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> createFences_semaphores() noexcept {
            fences.resize(frames_in_flight);
            waitSemaphore.resize(frames_in_flight);
            signalSemaphore.resize(imageCount);

            VkFenceCreateInfo fence{};
            fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            fence.pNext= nullptr;

            VkSemaphoreCreateInfo sem{};
            sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            sem.flags = 0;
            sem.pNext = nullptr;

            for(uint8_t i = 0 ; i < frames_in_flight ;i++){
                if(vkCreateFence(device , &fence , nullptr , &fences[i])  != VK_SUCCESS){
                    ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Fences"};
                    return std::unexpected(error);
                }
                if(vkCreateSemaphore(device , &sem , nullptr , &waitSemaphore[i]) != VK_SUCCESS){
                    ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create wait semaphores"};
                    return std::unexpected(error);
                }
            }

            for(uint32_t i = 0 ; i < imageCount ; i++){
                if(vkCreateSemaphore(device , &sem , nullptr , &signalSemaphore[i]) != VK_SUCCESS){
                    ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create signal semaphores"};
                    return std::unexpected(error);
                }
            }
            return {};
        }
 
        void clean(){
            for(auto& s : waitSemaphore)   if(s) vkDestroySemaphore(device , s , nullptr);
            for(auto& s : signalSemaphore) if(s) vkDestroySemaphore(device , s , nullptr);
            for(auto& f : fences)          if(f) vkDestroyFence(device , f , nullptr);

            waitSemaphore.clear();
            signalSemaphore.clear();
            fences.clear();

            if(sc) vkDestroySwapchainKHR(device , sc , nullptr);
        }

    public:

        Swapchain() = delete;

        struct SwapchainCreateInfo{
            VkPhysicalDevice pDevice;
            GLFWwindow* window;
            VkDevice device;
            VkSurfaceKHR surface;
        };

        Swapchain(const SwapchainCreateInfo& info){   
            if(!info.pDevice || !info.device || !info.window || !info.surface ) throw std::runtime_error("couldnt create the swapchain , since the info has nullptr");
            this->device = info.device;
            // creation of swapchain
            {
                auto result = init(info.surface , info.window , info.pDevice);
                if(result.has_value()){
                    const ErrorDataType res = {MOX_ERROR_TYPE_SUCCESS , "swap chain has been successfuly created"};
                    engineLogger(&res);
                }else{
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }

            {
                auto result = createFences_semaphores();
                if(result.has_value()){
                    const ErrorDataType res = {MOX_ERROR_TYPE_SUCCESS , "semaphores and fences have been successfuly created"};
                    engineLogger(&res);
                }else{
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }
        }


        Swapchain(Swapchain& input) = delete;
        Swapchain(Swapchain&& input){
            device = std::exchange(input.device , nullptr);
            sc = std::exchange(input.sc , nullptr);
            format = input.format;
            extent = input.extent;
            imageIndex = input.imageIndex;
            currentImage = input.currentImage;
            frameIndex = input.frameIndex;
            waitSemaphore = std::move(input.waitSemaphore);
            signalSemaphore = std::move(input.signalSemaphore);
            fences = std::move(input.fences);
            imageCount = input.imageCount;
        }

        Swapchain& operator=(Swapchain& input) = delete;
        Swapchain& operator=(Swapchain&& input){
            device = std::exchange(input.device , nullptr);
            sc = std::exchange(input.sc , nullptr);
            format = input.format;
            extent = input.extent;
            imageIndex = input.imageIndex;
            currentImage = input.currentImage;
            frameIndex = input.frameIndex;
            waitSemaphore = std::move(input.waitSemaphore);
            signalSemaphore = std::move(input.signalSemaphore);
            fences = std::move(input.fences);
            imageCount = input.imageCount;
            return *this;
        }

        VkSwapchainKHR sc = nullptr;
        VkSurfaceFormatKHR format;
        VkExtent2D extent;

        inline constexpr static uint32_t frames_in_flight = 3;
        uint32_t imageIndex = 0; // for RT (frame counter fed to the shaders)
        uint32_t currentImage = 0; // for frames in flight
        // counts every frame forever : the passes ping-pong their images by frameIndex % 2 ( currentImage
        // wraps at frames_in_flight = 3 , and 3 is not a multiple of 2 , so currentImage % 2 repeated an index )
        uint32_t frameIndex = 0;

        // the end of a frame : the next frame in flight and the next ping-pong index
        void nextFrame() noexcept{
            currentImage = (currentImage + 1) % frames_in_flight;
            frameIndex++;
        }

        uint32_t imageCount = 0;

        // no images for here , they are going to be in the Core

        std::vector<VkSemaphore> waitSemaphore{};
        std::vector<VkSemaphore> signalSemaphore{};
        std::vector<VkFence> fences{};

        ~Swapchain(){
            clean();
        }
    };
}