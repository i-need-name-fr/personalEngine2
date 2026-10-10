#pragma once


#include "../Image2/RenderImage2.hpp"
#include "../Image2/Textures2.hpp"
#include "../Image2/Skybox2.hpp"
#include "../Buffer2/GeneralBuffer.hpp"

namespace mox{

    enum class GraphRequiremenType{
        Undefined =0,
        Image = 1,
        Buffer = 2
    };

    inline static constexpr uint32_t moxGraphTypeImage = static_cast<uint32_t>(GraphRequiremenType::Image);
    inline static constexpr uint32_t moxGraphTypeBuffer = static_cast<uint32_t>(GraphRequiremenType::Buffer);
    inline static constexpr uint32_t moxGraphTypeUndefined = static_cast<uint32_t>(GraphRequiremenType::Undefined);


    struct GraphParameterInfo{
        uint32_t type{moxGraphTypeUndefined};
        struct{
            ImageType type;
        } imageSpecification;
        std::string name{"undefined"};
    };

    
    struct GraphParameterSource{
        std::string name;
        void* location{nullptr};
    };

    // =================================== INFO ===================================
    // GraphPass2 UPDATE:
    // 10.10.2026
    // GraphPass2 still has the same kinda functions
    // however some are new functions
    // 
    // NOTE: brend new style
    // we dont have to input many buffers and images in create info between all the graph passes
    // we simply have to add them in graphRender , and it will transfer them all along th graphPasses
    // each graphPass will have set of pointers to their parameters
    // 
    // first is requestExectution() is it a function that defines whether we should execute this graph or not
    // postProcessGraphRenderLocations - function that provides pointer to the required data parameters 
    // 
    // each graph Pass will translate the image to its required Access and so on , there is literally no need for graph Render to do anything
    // since now the Image2 and Buffer2 do it automatically
    // the only thing that Graph Render does , is providing the locations of paramaters for each pass
    // removing need for providing anything in createInfo
    // =================================== END ===================================

    class GraphPass2{
    public:

        GraphPass2() = default;
        GraphPass2(GraphPass2& input) = delete;
        GraphPass2(const GraphPass2& input) = delete;
        GraphPass2& operator=(GraphPass2& input ) = delete;
        GraphPass2& operator=(const GraphPass2& input) = delete;
        
        GraphPass2(GraphPass2&& input) noexcept{

        }

        GraphPass2& operator=(GraphPass2&& input ) noexcept{

            return *this;
        }

        virtual std::vector<GraphParameterInfo> acquireRequirements() noexcept = 0;
        [[nodiscard]] [[gnu::hot]] virtual ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept = 0;
        [[nodiscard]] [[gnu::hot]] virtual const bool requestExecution() const noexcept = 0;
        [[nodiscard]] virtual const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept = 0;
        [[nodiscard]] [[gnu::hot]] virtual const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept = 0;

        virtual ~GraphPass2(){}

    };

    using pGraphPass2 = std::shared_ptr<GraphPass2>;
}