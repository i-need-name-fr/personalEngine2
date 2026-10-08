#pragma once 

#include "RenderImage2.hpp"

namespace mox{
    // layer count should be more then 1 , since if its the only one , it belongs to main image view from RenderImage or Skybox
    
    template<uint32_t T>
    requires(T >= 2)
    class ViewPerLayer final{
    public:

        ViewPerLayer() = delete;
        ViewPerLayer(ViewPerLayer& input) = delete;
        ViewPerLayer(const ViewPerLayer& input) = delete;
        ViewPerLayer& operator=(ViewPerLayer& input) = delete;
        ViewPerLayer& operator=(const ViewPerLayer& input) = delete;

        ViewPerLayer(ViewPerLayer&& input) noexcept{
            if(views == input.views) return ;
            processCopy(std::move(input));
        }

        ViewPerLayer& operator=(ViewPerLayer&& input) noexcept{
            if(views == input.views) return *this;
            processCopy(std::move(input));
            return *this;
        }

        VkImageView& operator[](const uint32_t index){
            if(index - 1 >= views.size()) throw std::runtime_error(std::format("couldnt get view per layer index -> {} , since its out of range , size -> {}", index - 1 , views.size()));
            return views[index - 1];
        }

        struct CreateInfo{
            VulkanContext* context = nullptr;
            VkImage image = nullptr;
            VkImageAspectFlags aspect;
            VkImageViewType viewType;
            VkImageViewCreateFlags flags;
            VkFormat format;
        };


        
        ViewPerLayer(const CreateInfo& info){

            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypeQueueManager , info.context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }

            device = info.context->logicalDevice->device;

            {
                const auto result = createViews(info);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }
        }

        VkImageMemoryBarrier2 translateImage(const VkAccessFlagBits2 access ,const VkPipelineStageFlags2 stage , const uint32_t layerIndex) {

            if(layerIndex < 1) throw std::invalid_argument("couldnt create image barrier,  since the layer index is below minimum , which is 1 , layer 0 belongs to parent image view");
            if(layerIndex > views.size()) throw std::invalid_argument("couldnt create image barrier,  since the layer index is bigger than the count of the views");

            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.image = inputImage;
            barrier.pNext = nullptr;
            barrier.srcAccessMask = accesses[layerIndex -1];
            barrier.dstAccessMask = access;
            barrier.srcStageMask = stages[layerIndex -1];
            barrier.dstStageMask = stage;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.subresourceRange.aspectMask = aspect;
            barrier.subresourceRange.baseArrayLayer = layerIndex;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.levelCount = 1;

            accesses[layerIndex - 1] = access;
            stages[layerIndex - 1] = stage;

            return barrier;
        }

        VkImageMemoryBarrier2 translateImageUndefined(const VkAccessFlagBits2 access ,const VkPipelineStageFlags2 stage  , const uint32_t layerIndex){

            if(layerIndex < 1) throw std::invalid_argument("couldnt create image barrier,  since the layer index is below minimum , which is 1 , layer 0 belongs to parent image view");
            if(layerIndex > views.size()) throw std::invalid_argument("couldnt create image barrier,  since the layer index is bigger than the count of the views");

            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.image = inputImage;
            barrier.pNext = nullptr;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = access;
            barrier.srcStageMask = stages[layerIndex -1];
            barrier.dstStageMask = stage;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.subresourceRange.aspectMask = aspect;
            barrier.subresourceRange.baseArrayLayer = layerIndex;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.levelCount = 1;

            accesses[layerIndex - 1] = access;
            stages[layerIndex - 1] = stage;

            return barrier;
        }

        VkImageMemoryBarrier2 translateImageCombined(const VkAccessFlagBits2 access ,const VkPipelineStageFlags2 stage ,const uint32_t layerCount, const uint32_t layerIndex){
            if(layerCount < 1) throw std::invalid_argument("couldnt create image barrier,  since the layer count is below minimum , which is 1");
            if(layerIndex < 1) throw std::invalid_argument("couldnt create image barrier,  since the layer index is below minimum , which is 1 , layer 0 belongs to parent image view");
            if(layerIndex - 1 + layerCount > views.size()) throw std::invalid_argument("... range is bigger than the count of the views");

            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.image = inputImage;
            barrier.pNext = nullptr;
            barrier.dstAccessMask = access;
            barrier.dstStageMask = stage;

            VkAccessFlags2 srcA = 0; VkPipelineStageFlags2 srcS = 0;
            for(uint32_t k = 0 ; k < layerCount ; k++){
                srcA |= accesses[layerIndex - 1 + k];
                srcS |= stages  [layerIndex - 1 + k];
            }
            barrier.srcAccessMask = srcA; barrier.srcStageMask = srcS;

            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.subresourceRange.aspectMask = aspect;
            barrier.subresourceRange.baseArrayLayer = layerIndex;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.layerCount = layerCount;
            barrier.subresourceRange.levelCount = 1;


            for(uint32_t layer = 0 ; layer < layerCount ; layer++){
                const uint32_t currentIndex = layer + layerIndex - 1;
                if(currentIndex >= accesses.size()) break;
                accesses[currentIndex] = access;
                stages[currentIndex] = stage;
            }

            return barrier;
        }

        VkAccessFlags2 getAccessViewAt(const uint32_t layerIndex) const noexcept{
            return accesses[layerIndex];
        }

        VkPipelineStageFlags2 getPipelineStageAt(const uint32_t layerIndex) const noexcept{
            return stages[layerIndex];
        }

        // index should start with 1 , since the 0 belongs to input image 
        VkImageView& at(const uint32_t index){
            if(index -1 >= views.size()) throw std::runtime_error(std::format("couldnt get view per layer index -> {} , since its out of range , size -> {}", index -1 , views.size()));
            return views[index -1];
        }

        const uint64_t size() const noexcept{
            return views.size();
        }

        ~ViewPerLayer(){processDelete();}

    private:

        VkDevice device = nullptr;

        VkImage inputImage = nullptr;
        std::vector<VkImageView> views{};
        VkImageAspectFlags aspect{VK_ACCESS_NONE};
        std::vector<VkAccessFlags2> accesses{};
        std::vector<VkPipelineStageFlags2> stages{};

        [[nodiscard]] ErrorDataOutput<void> createViews(const CreateInfo &input) noexcept{

            views.resize(T);
            // we start from 1, since the 0 belongs to main view or Render image or Skybox
            for(uint32_t i = 1 ; i <= T ; i++){
                VkImageViewCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                info.viewType = input.viewType;
                info.image = input.image;
                info.flags = input.flags;
                info.format = input.format;
                info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
                info.pNext = nullptr;
                info.subresourceRange.aspectMask = input.aspect;
                info.subresourceRange.baseMipLevel = 0;
                info.subresourceRange.levelCount = 1;
                info.subresourceRange.layerCount = 1;
                info.subresourceRange.baseArrayLayer = i;

                const auto result = vkCreateImageView(device , &info ,nullptr,  &views[i -1]);
                if(result != VK_SUCCESS){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create image view for view per layer , layer -> {} , error ID -> {}" , i, static_cast<int32_t>(result))});
                }
            }

            aspect = input.aspect;
            accesses = std::move(std::vector<VkAccessFlags2>(T , VK_ACCESS_NONE));
            stages = std::move(std::vector<VkPipelineStageFlags2>(T , VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT));
            inputImage = input.image;

            cmdFunctionType function = [&](VkCommandBuffer& cmd) ->void {
                auto barrier = translateImageCombined(VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT , views.size() , 1 );

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.bufferMemoryBarrierCount = 0;
                dep.imageMemoryBarrierCount = 1;
                dep.pMemoryBarriers = nullptr;
                dep.pBufferMemoryBarriers = nullptr;
                dep.pImageMemoryBarriers = &barrier;
                dep.memoryBarrierCount = 0;
                dep.pNext = nullptr;
                dep.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &dep);
            };

            input.context->queueManager->Begin();

            input.context->queueManager->recordCmd(function ,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            {
                const auto result = input.context->queueManager->End();
                if(!result.has_value()) return result;
            }

            return {};
        }

        void processCopy(ViewPerLayer&& input) noexcept{
            processDelete();
            device = std::exchange(input.device , nullptr);
            aspect = input.aspect;
            accesses = std::move(input.accesses);
            stages = std::move(input.stages);
            inputImage = std::exchange(input.inputImage , nullptr);
            views =std::move(input.views);
        }

        void processDelete() noexcept{
            if(device && !views.empty()){
                for(auto& view : views){
                    if(!view) continue;
                    vkDestroyImageView(device , view , nullptr);
                }
            }
            views.clear();
            device = nullptr;
            inputImage = nullptr;
        }
    };
    
    template<uint32_t T>
    requires(T >= 2)
    using pViewPerLayer = std::shared_ptr<ViewPerLayer<T>>;
}