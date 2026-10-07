#pragma once

#include "LogicalDevice.hpp"
#include "SwapChain.hpp"
#include <functional>
#include <memory>
#include <random>
#include <shared_mutex>

namespace mox{

    class threadQueue final{
    public:

        threadQueue() = delete;
        threadQueue(threadQueue& input) = delete;
        threadQueue& operator=(threadQueue& input) = delete;

        threadQueue(threadQueue&& input) noexcept {
            copyThread(std::move(input));
        }

        threadQueue& operator=(threadQueue&& input) noexcept {
            copyThread(std::move(input));
            return *this;
        }

        struct CreateInfo{
            VkDevice device = nullptr;
            VkPhysicalDevice pDevice = nullptr;
            VkSurfaceKHR surface = nullptr;

            uint32_t index = 0;
        };

        threadQueue(const CreateInfo &info){
            if(!info.device || !info.pDevice || !info.surface ){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create thread queue , since the info has null data"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(info.index > 16){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create thread queue , since the index for thread queue is bigger than amount of queues"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            device = info.device;
            acquireQueue(info);
            {
                const auto result = createPool(info);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }

            {
                const auto result = createSemaphoreFence();
                if(!result.has_value()){
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }
        }

        ~threadQueue(){
            cleanUp();
        }

        VkQueue queue = nullptr;
        VkCommandPool pool = nullptr;

        VkSemaphore signalSemaphore;

        [[nodiscard]] ErrorDataOutput<void> recordBuffer(std::function<void(VkCommandBuffer&)> function) noexcept {

            if(submitted){
                vkWaitForFences(device , 1 , &fence , VK_TRUE , UINT64_MAX);
                vkResetFences(device , 1 , &fence);

                for(auto& index : cmdIndices){
                    vkResetCommandBuffer(cmds[index] , 0);
                    checkCmd[index] = false;
                }
                cmdIndices.clear();
                submitted = false;
            }

            uint32_t cmdIndex = 0;
            {
                const auto result = acquireCMD();
                if(result.has_value()){
                    cmdIndex = result.value();
                }else{
                    return std::unexpected(result.error());
                }
            }

            VkCommandBufferBeginInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            info.pInheritanceInfo = nullptr;
            info.flags = 0;
            info.pNext = nullptr;

            {
                const auto result = vkBeginCommandBuffer(cmds[cmdIndex] , &info);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to begin command buffer , error ID -> {}" , static_cast<int32_t>(result))});
                }
            }

            function(cmds[cmdIndex]);

            {
                const auto result = vkEndCommandBuffer(cmds[cmdIndex]);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to end command buffer , error ID -> {}" , static_cast<int32_t>(result))});
                }
            }
            return {};
        }
        
        [[nodiscard]] ErrorDataOutput<void> executeThread(bool needsSignal,
                        VkPipelineStageFlags* dstStages,
                        VkSemaphore* waitSemaphores,
                        uint32_t waitCount) noexcept
        {
            [[unlikely]] if (submitted){
                const ErrorDataType error{MOX_ERROR_TYPE_WARNING , "failed to execute the thread , since the queue is already submitted"};
                return std::unexpected(error);
            }
            [[unlikely]] if (cmdIndices.empty()){
                const ErrorDataType error{MOX_ERROR_TYPE_WARNING , "failed to execute the thread , since there is no task to do"};
                return std::unexpected(error);

            }
            [[unlikely]] if (waitCount > 0 && (waitSemaphores == nullptr || dstStages == nullptr)){
                const ErrorDataType error{MOX_ERROR_TYPE_WARNING , "failed to execute the thread , since the data given is nullptr"};
                return std::unexpected(error);

            }


            std::vector<VkCommandBuffer> executableCMD;
            executableCMD.reserve(cmdIndices.size());
            for (auto idx : cmdIndices) executableCMD.push_back(cmds[idx]);

            VkSubmitInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            info.commandBufferCount = static_cast<uint32_t>(executableCMD.size());
            info.pCommandBuffers = executableCMD.data();

            if (waitCount > 0) {
                info.waitSemaphoreCount = waitCount;
                info.pWaitSemaphores   = waitSemaphores;
                info.pWaitDstStageMask = dstStages;
            }

            if (needsSignal) {
                info.signalSemaphoreCount = 1;
                info.pSignalSemaphores = &signalSemaphore;
            }

            const auto result =vkQueueSubmit(queue, 1, &info, fence);
            if(result != VK_SUCCESS){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to submit the queue , error ID -> {}" , static_cast<int32_t>(result))});
            }
            submitted = true;

            return {};
        }

        void waitForQueue(){
            vkWaitForFences(device , 1 , &fence , VK_TRUE , UINT64_MAX);
        }

    private:

        [[gnu::cold]] inline void copyThread(threadQueue&& input) noexcept {

            queue = std::exchange(input.queue , nullptr);
            pool = std::exchange(input.pool , nullptr);
            device = std::exchange(input.device , nullptr);
            cmds = std::move(input.cmds);
            cmdIndices = std::move(input.cmdIndices);
            checkCmd = std::move(input.checkCmd);
            fence = std::exchange(input.fence , nullptr);
            signalSemaphore = std::exchange(input.signalSemaphore , nullptr);
            submitted = std::move(input.submitted);
        }

        inline void cleanUp(){
            if(device){
                if(queue) vkQueueWaitIdle(queue);
                if(fence) vkDestroyFence(device , fence , nullptr);
                if(signalSemaphore) vkDestroySemaphore(device , signalSemaphore, nullptr);
                if(pool) vkDestroyCommandPool(device , pool , nullptr);
            }
        }

        ErrorDataOutput<void> createSemaphoreFence(){
            VkSemaphoreCreateInfo info{};
            info.sType=  VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            info.flags = 0;
            info.pNext = nullptr;

            const auto result = vkCreateSemaphore(device , &info , nullptr , &signalSemaphore);
            if(result != VK_SUCCESS){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create semaphore for thread queue , error ID -> {}" , static_cast<int32_t>(result))};
                return std::unexpected(error);
            }

            {
                VkFenceCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                info.flags = 0;
                info.pNext = nullptr;

                const auto result = vkCreateFence(device , &info , nullptr , &fence);
                if(result != VK_SUCCESS){
                    const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create fence for thread queue , error ID -> {}" , static_cast<int32_t>(result))};
                    return std::unexpected(error);
                }                
            }
            return {};
        }

        ErrorDataOutput<void> createPool(const CreateInfo& input){

            DetailsCreateInfo data{};
            data.device = input.pDevice;
            data.surface = input.surface;

            auto details = queueFamilyDetails::queryQueueFamilyDetails(data);

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
            return {};
        }

        void acquireQueue(const CreateInfo& input){
            DetailsCreateInfo data{};
            data.device = input.pDevice;
            data.surface = input.surface;

            auto details = queueFamilyDetails::queryQueueFamilyDetails(data);
            vkGetDeviceQueue(device, details.graphics.value() , input.index , &queue );
        }

        ErrorDataOutput<int32_t> acquireCMD() noexcept{
            if(cmds.empty()){
                VkCommandBuffer cmd = nullptr;

                VkCommandBufferAllocateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                info.pNext = nullptr;
                info.commandPool = pool;
                info.commandBufferCount = 1;
                info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

                const auto result = vkAllocateCommandBuffers(device , &info ,&cmd );
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to allocate the cmd , error ID -> {}" , static_cast<int32_t>(result))});
                }

                cmds.push_back(std::move(cmd));

                checkCmd.push_back(true);
                cmdIndices.push_back(0);

                return 0;

            }else{

                for(uint32_t i = 0 ; i < checkCmd.size() ; i++){
                    if(!checkCmd[i]){
                        checkCmd[i] = true;
                        cmdIndices.push_back(i);
                        return i;
                    }
                }

                VkCommandBuffer cmd = nullptr;

                VkCommandBufferAllocateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                info.pNext = nullptr;
                info.commandPool = pool;
                info.commandBufferCount = 1;
                info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

                const auto result = vkAllocateCommandBuffers(device , &info ,&cmd );
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to allocate the cmd , error ID -> {}" , static_cast<int32_t>(result))});
                }

                const uint32_t lastIndex = cmds.size();

                cmds.push_back(cmd);
                checkCmd.push_back(true);
                cmdIndices.push_back(lastIndex);

                return lastIndex;
            }
        }

        VkDevice device = nullptr;

        VkFence fence = nullptr;

        std::vector<VkCommandBuffer> cmds{};

        std::vector<uint32_t> cmdIndices{};
        std::vector<uint8_t> checkCmd{};

        uint8_t submitted = false;

    };

    using pThreadQueue = std::shared_ptr<threadQueue>;
    using cmdFunctionType = std::function<void(VkCommandBuffer&)>;

    class QueueManager final{
    public:

        QueueManager() = delete;
        QueueManager(QueueManager& input) = delete;
        QueueManager& operator=(QueueManager& input) = delete;

        QueueManager(QueueManager&& input) noexcept {
            processTheCopy(std::move(input));
        }

        QueueManager& operator=(QueueManager&& input) noexcept {
            processTheCopy(std::move(input));
            return *this;
        }

        ~QueueManager(){
            cleanQueueManager();
        }

        struct CreateInfo{

            VkDevice device = nullptr;
            VkSurfaceKHR surface=  nullptr;
            VkPhysicalDevice pDevice = nullptr;

            uint32_t firstIndex;
            uint32_t threadsCount;
        };

        QueueManager(const CreateInfo &info){
            if(!info.device){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the queue manager , since the logical device is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(!info.pDevice){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the queue manager , since the physical device is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            if(!info.surface){
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the queue manager , since the surface is nullptr"};
                engineLogger(&error);
                THROW_MESSAGE;
            }
            device = info.device;
            {
                const auto result = this->createAllQueues(info);
                if(!result.has_value()){
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }
        }

        void Begin(){
            if(!queuesIndices.empty()){
                for(auto& index : queuesIndices){
                    queues[index]->waitForQueue();
                }
                queuesIndices.clear();
            }

            started = true;
        }

        void recordCmd(cmdFunctionType &function , VkPipelineStageFlags dstStage) noexcept{
            if(!started) Begin();

            work.push_back({function , dstStage});

        }


        ErrorDataOutput<void> End(){
            if(!started || work.empty()){
                return {};
            }

            const uint32_t remainder = work.size() % queuesCount;  // было workColon
            const uint32_t fullRows  = work.size() / queuesCount;  // было workRow

            auto processGroup = [&](uint32_t baseIdx, uint32_t groupSize) ->ErrorDataOutput<void> {
                for (uint32_t i = 0; i < groupSize; i++) {
                    const auto result = queues[i]->recordBuffer(work[baseIdx + i].first);
                    if(!result.has_value()){
                        return std::unexpected(result.error());
                    }
                }

                std::vector<VkSemaphore>          waits;
                std::vector<VkPipelineStageFlags> stages;
                if (groupSize > 1) {
                    waits.reserve(groupSize - 1);
                    stages.reserve(groupSize - 1);
                    for (uint32_t j = 0; j + 1 < groupSize; j++) {
                        waits.push_back(queues[j]->signalSemaphore);
                        stages.push_back(work[baseIdx + j].second);
                    }
                }

                for (uint32_t i = 0; i < groupSize; i++) {
                    if (i == groupSize - 1) {
                        const auto result =queues[i]->executeThread(
                            false,
                            stages.data(),
                            waits.data(),
                            static_cast<uint32_t>(waits.size())
                        );
                        if(!result.has_value()){
                            return std::unexpected(result.error());
                        }
                    } else {
                        const auto result =queues[i]->executeThread(true, nullptr, nullptr, 0);
                        if(!result.has_value()){
                            return std::unexpected(result.error());

                        }
                    }
                }
                return {};
            };

            for (uint32_t row = 0; row < fullRows; row++) {
                if (row > 0) {
                    queues[queuesCount - 1]->waitForQueue();
                }
                const auto result = processGroup(row * queuesCount, queuesCount);
                        if(!result.has_value()){
                            return std::unexpected(result.error());
                        }
            }

            if (remainder) {
                if (fullRows > 0) {
                    queues[queuesCount - 1]->waitForQueue();
                }
                const auto result = processGroup(fullRows * queuesCount, remainder);
                if(!result.has_value()){
                    return std::unexpected(result.error());
                }
            }

            if (fullRows > 0 || remainder > 0) {
                const uint32_t lastUsedQueue = (remainder > 0) ? (remainder - 1)
                                                            : (queuesCount - 1);
                queues[lastUsedQueue]->waitForQueue();
            }

            queuesIndices.clear();
            const uint32_t usedCount = (fullRows == 0) ? remainder : queuesCount;
            for (uint32_t i = 0; i < usedCount; i++) {
                queuesIndices.push_back(i);
            }

            started = false;
            work.clear();
            return {};
        }

    private:

        VkDevice device = nullptr;

        inline void cleanQueueManager(){
            queues.clear();
        }

        void processTheCopy(QueueManager&& input) noexcept{
            {
                std::lock_guard<std::shared_mutex> guard(mutex);
                queues = std::move(input.queues);
                queuesIndices = std::move(input.queuesIndices);

                work = std::move(input.work);
                started = input.started;              // без move
                queuesCount = input.queuesCount;      // без move
                started = input.started;
            }
        }

        [[nodiscard]] ErrorDataOutput<void> createAllQueues(const CreateInfo &info) noexcept{

            try{
                for(uint32_t i = info.firstIndex ; i < info.threadsCount ; i++){
                    threadQueue::CreateInfo data{};
                    data.index = i;
                    data.device = info.device;
                    data.pDevice = info.pDevice;
                    data.surface = info.surface;

                    auto queue = std::make_shared<threadQueue>(data);
                    queues.push_back(std::move(queue));
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create all queues for threadQueue"});
            }
            queuesCount = queues.size();
            return {};
        }

        uint32_t queuesCount;

        std::vector<pThreadQueue> queues{};
        std::vector<uint32_t> queuesIndices{};
        std::vector<std::pair<cmdFunctionType , VkPipelineStageFlags>> work{};
        std::shared_mutex mutex;
        bool started = false;
    };


    using pQueueManager = std::shared_ptr<QueueManager>;
}