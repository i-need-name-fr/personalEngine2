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

    // the address of a field of the pass : bindSources fills the field itself
    template<typename T>
    struct GraphSlot{
        std::string_view name;
        T** target;
    };

    // idea here is simple to delete all attachments

    namespace Graph
    {
        typedef uint32_t AttributeType;

        inline static const AttributeType AttributeUndefined = 0;
        inline static const AttributeType AttributeBuffer2 = 1;
        inline static const AttributeType AttributeRenderImage2 = 2;
        inline static const AttributeType AttributeSkybox2 = 3;
        inline static const AttributeType AttributeTexture2 = 4;
        inline static const AttributeType AttributeTLAS2 = 5;


        struct Attribute
        {
            AttributeType type;
            void* source;
        };

        inline static ErrorDataOutput<void> processBindings(const std::vector<GraphParameterSource> &sources , std::unordered_map<std::string , Graph::Attribute> &data) noexcept
        {   
            uint32_t processedData_ = 0ul;
            for(auto& source : sources)
            {
                if(!source.location)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("while processing the data , Source : {} is a nullptr"  , source.name)});
                }
                if(!data.contains(source.name))
                {
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to process Source : {} , since the GraphPass2 didnt require for it, MIGHT CAUSE TROUBLE" , source.name)};
                    engineLogger(error);
                    continue;
                }

                data[source.name].source = source.location;
                processedData_++;
            }

            if(processedData_ != data.size())
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process data for GraphPass2 , since some sources were not provided"});
            }

            return {};
        }
    }

#define _castToRenderImage2(data) std::bit_cast<RenderImage2*>(data)
#define _castToSkybox2(data) std::bit_cast<Skybox2*>(data)
#define _castToBufferBase2(data) std::bit_cast<BufferBase2*>(data)
#define _castToTexture2(data) std::bit_cast<Texture2*>(data)


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
    // that function also creates the pipelines and so on 
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
            _attributes = std::move(input._attributes);
        }

        GraphPass2& operator=(GraphPass2&& input ) noexcept{
            if(this != &input) _attributes = std::move(input._attributes);
            return *this;
        }

        virtual std::vector<GraphParameterInfo> acquireRequirements() noexcept = 0;
        [[nodiscard]] [[gnu::hot]] virtual ErrorDataOutput<void> executePass(VkCommandBuffer& cmd) noexcept = 0;
        [[nodiscard]] [[gnu::hot]] virtual const bool requestExecution() const noexcept = 0;
        [[nodiscard]] virtual const ErrorDataOutput<void> postProcessGraphRenderLocations(std::vector<GraphParameterSource> sources) noexcept = 0;
        [[nodiscard]] [[gnu::hot]] virtual const ErrorDataOutput<void> translateBarriers(VkCommandBuffer& cmd) noexcept = 0;

        virtual ~GraphPass2(){}

        // true when the pass has got its resources and has created its pipeline : the graph does not link such a pass again ,
        // so a pass that is added to several paths is not created several times
        [[nodiscard]] bool isInitialized() const noexcept{
            return completedInit;
        }

    protected:

        uint8_t completedInit = false;

        // gives the pointers that the graph found to the fields of the pass ( the slots hold the ADDRESS of a field ) .
        // every slot has to be filled : a source that wasnt requested , a null location and a slot that got nothing are errors
        template<typename... Ts>
        [[nodiscard]] ErrorDataOutput<void> bindSources(const std::vector<GraphParameterSource>& sources , const GraphSlot<Ts>&... slots) noexcept{
            // a second call ( the graph was built again ) must not keep the old pointers
            ((*slots.target = nullptr) , ...);

            for(const auto& source : sources){
                if(!source.location){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process graph source named -> {} , since its location is nullptr" , source.name)});
                }
                if(!(bindSlot(source , slots) || ...)){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process graph source named -> {} , since it wasnt requested" , source.name)});
                }
            }

            std::string_view missing{};
            const auto checkSlot = [&missing](const auto& slot){
                if(!*slot.target && missing.empty()) missing = slot.name;
            };
            (checkSlot(slots) , ...);

            if(!missing.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process graph sources , since the source named -> {} wasnt given" , missing)});
            }
            return {};
        }

        // ======================= the attributes =======================
        // the pass has NO pointer fields for the images and the buffers of the graph : everything that it requires ( acquireRequirements ) lives in _attributes , by the name .
        // postProcessGraphRenderLocations calls bindAttributes( sources ) once ; after it image( "name" ) / buffer( "name" ) give the object . the pairs of the ping pong have
        // the names "base_0" and "base_1" : image( "base" , index ) takes the right one
        [[nodiscard]] ErrorDataOutput<void> bindAttributes(const std::vector<GraphParameterSource>& sources) noexcept{
            _attributes.clear();
            for(const auto& requirement : acquireRequirements()){
                Graph::Attribute attribute{};
                attribute.source = nullptr;
                if(requirement.type == moxGraphTypeBuffer){
                    attribute.type = Graph::AttributeBuffer2;
                }else if(requirement.type == moxGraphTypeImage){
                    switch(requirement.imageSpecification.type){
                        case ImageType::Skybox2: attribute.type = Graph::AttributeSkybox2; break;
                        case ImageType::Texture2: attribute.type = Graph::AttributeTexture2; break;
                        default: attribute.type = Graph::AttributeRenderImage2; break;
                    }
                }else{
                    attribute.type = Graph::AttributeUndefined;
                }
                _attributes.emplace(requirement.name , attribute);
            }
            return Graph::processBindings(sources , _attributes);
        }

        [[nodiscard]] void* attributeSource(const std::string_view name) const noexcept{
            const auto it = _attributes.find(std::string(name));
            return it == _attributes.end() ? nullptr : it->second.source;
        }

        [[nodiscard]] static std::string pingPongName(const std::string_view base , const uint32_t index) noexcept{
            std::string name(base);
            name += '_';
            name += static_cast<char>('0' + (index & 1u));
            return name;
        }

        [[nodiscard]] RenderImage2* image(const std::string_view name) const noexcept{ return _castToRenderImage2(attributeSource(name)); }
        [[nodiscard]] RenderImage2* image(const std::string_view base , const uint32_t index) const noexcept{ return image(pingPongName(base , index)); }
        [[nodiscard]] Skybox2* skybox(const std::string_view name) const noexcept{ return _castToSkybox2(attributeSource(name)); }
        [[nodiscard]] BufferBase2* buffer(const std::string_view name) const noexcept{ return _castToBufferBase2(attributeSource(name)); }
        [[nodiscard]] BufferBase2* buffer(const std::string_view base , const uint32_t index) const noexcept{ return buffer(pingPongName(base , index)); }

        void clearAttributes() noexcept{ _attributes.clear(); }

    private:

        template<typename T>
        static bool bindSlot(const GraphParameterSource& source , const GraphSlot<T>& slot) noexcept{
            if(source.name != slot.name) return false;
            *slot.target = std::bit_cast<T*>(source.location);
            return true;
        }

        std::unordered_map<std::string , Graph::Attribute> _attributes{};
    };

    using pGraphPass2 = std::shared_ptr<GraphPass2>;
}