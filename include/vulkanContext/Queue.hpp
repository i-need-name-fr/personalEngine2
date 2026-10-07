#pragma once

#include "LogicalDevice.hpp"
#include "SwapChain.hpp"
#include <memory>

namespace mox{

    class Queue{
    public:

        struct CreateInfo{
            VkDevice device = nullptr;
            VkPhysicalDevice pDevice = nullptr;
            VkSurfaceKHR surface = nullptr;

            uint32_t countIndex = 0;
            bool needsCmd = false;
        };

        Queue() = delete;
        Queue(Queue& input) = delete;
        Queue& operator=(Queue& input) = delete;

        Queue(Queue&& input){
            processCopy(std::move(input));
        }

        Queue& operator=(Queue&& input){
            processCopy(std::move(input));
            return *this;
        }

        virtual ~Queue(){
            cleanQueue();
        }

        Queue(const CreateInfo& info){
            this->device = info.device;

            const auto result = initTheQueue(info);
            if(result.has_value()){
            }else{
                engineLogger(&result);
                THROW_MESSAGE;
            }
        }

        VkCommandPool pool = nullptr;
        VkQueue queue = nullptr;
        
        std::vector<VkCommandBuffer> cmds{};

    protected:

        virtual void cleanQueue(){
            if(device){
                if(!cmds.empty()) vkFreeCommandBuffers(device , pool , static_cast<uint32_t>(cmds.size()) , cmds.data());
                if(pool) vkDestroyCommandPool(device,  pool , nullptr);
            }
        }

        void processCopy(Queue&& input){
            device = std::exchange(input.device,  nullptr);
            pool = std::exchange(input.pool , nullptr);
            queue = std::exchange(input.queue , nullptr);
            cmds = std::move(input.cmds);
        }

        [[gnu::cold]] [[nodiscard]] ErrorDataOutput<void> initTheQueue(const CreateInfo& info){

            [[unlikely]] if(!info.device){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the queue , since the info is nullptr"};
                return std::unexpected(error);
            }

            [[unlikely]] if(!info.pDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the queue , since the info is nullptr"};
                return std::unexpected(error);
            }

            [[unlikely]] if(!info.surface){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the queue , since the info is nullptr"};
                return std::unexpected(error);
            }

            DetailsCreateInfo data{};
            data.device = info.pDevice;
            data.surface = info.surface;

            auto details = queueFamilyDetails::queryQueueFamilyDetails(data);

            vkGetDeviceQueue(device , details.graphics.value() , info.countIndex , &queue);

            [[unlikely]] if(!queue){
                const ErrorDataType error{MOX_ERROR_TYPE_CRITICAL , "failed to create the queue for some reason, the queue in init is nullptr"};
                return std::unexpected(error);
            }

            {
                VkCommandPoolCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                info.pNext = nullptr;
                info.queueFamilyIndex = details.graphics.value();

                const auto result = vkCreateCommandPool(device , &info , nullptr , &pool);
                [[unlikely]] if(result != VK_SUCCESS){
                    const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create command pool,  error ID -> {}" , static_cast<int32_t>(result))};
                    return std::unexpected(error);
                }
            }

            // since this is a typical queue 
            // we will also make the command buffers here , and allocate them to 3
            cmds.resize(Swapchain::frames_in_flight);

            if(info.needsCmd){
                {
                    VkCommandBufferAllocateInfo info{};
                    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                    info.pNext = nullptr;
                    info.commandPool = pool;
                    info.commandBufferCount = Swapchain::frames_in_flight;
                    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

                    const auto result = vkAllocateCommandBuffers(device , &info , cmds.data());
                    [[unlikely]] if(result != VK_SUCCESS){
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create command buffers , error ID -> {}" , static_cast<int32_t>(result))};
                        return std::unexpected(error);
                    }
                }
            }

            return {};
        }

        VkDevice device = nullptr;
    };

    class QueueLane final : public Queue{
    public:

        QueueLane() = delete;
        QueueLane(QueueLane& input) = delete;
        QueueLane(QueueLane&& input) noexcept : Queue(std::move(input)){
            signalSemaphore = std::move(input.signalSemaphore);
            fences = std::move(input.fences);
        }

        QueueLane& operator=(QueueLane& input) = delete;
        QueueLane& operator=(QueueLane&& input) noexcept {
            processCopy(std::move(input));
            signalSemaphore = std::move(input.signalSemaphore);
            fences = std::move(input.fences);
            return *this;
        }

        virtual ~QueueLane(){
            if(device){
                for(auto& sem : signalSemaphore){
                    if(sem) vkDestroySemaphore(device , sem , nullptr);
                }
                for(auto& fence : fences){
                    if(fence) vkDestroyFence(device , fence , nullptr);
                }
            }
        }

        std::vector<VkSemaphore> signalSemaphore{};
        std::vector<VkFence> fences{};

        QueueLane(const CreateInfo& info) : Queue(info){
            const auto result = createSemaphoresAndFences();
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

    private:

        [[gnu::cold]] [[nodiscard]] ErrorDataOutput<void> createSemaphoresAndFences() noexcept{

            VkFenceCreateInfo fence{};
            fence.flags = 0;
            fence.pNext = nullptr;
            fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

            VkSemaphoreCreateInfo sem{};
            sem.flags = 0;
            sem.pNext = nullptr;
            sem.sType  = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            signalSemaphore.resize(Swapchain::frames_in_flight);
            fences.resize(Swapchain::frames_in_flight);

            for(uint8_t i = 0 ; i < Swapchain::frames_in_flight ; i++){
                {
                    const auto result = vkCreateFence(device , &fence , nullptr , &fences[i]);
                    if(result != VK_SUCCESS){
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create fence , error ID -> {}" , static_cast<int32_t>(result))};
                        return std::unexpected(error);
                    }
                }

                {
                    const auto result = vkCreateSemaphore(device , &sem , nullptr , &signalSemaphore[i]);
                    if(result != VK_SUCCESS){
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create semaphores , error ID -> {}" , static_cast<int32_t>(result))};
                        return std::unexpected(error);
                    }
                }

            }

            return {};
        }

    };

    using pQueueLane = std::shared_ptr<QueueLane>;
    using pQueue = std::shared_ptr<Queue>;    
}