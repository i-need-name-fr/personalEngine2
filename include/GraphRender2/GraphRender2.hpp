#pragma once

#include "GraphPass2.hpp"
#include "../Image2/FrameTargets.hpp"

namespace mox{
    // =================================== INFO ===================================
    // ---------------------------- GRAPH RENDER2 -------------------------
    // GENERAL INFO:
    // now GraphRender provides with all attachments
    // =================================== END ===================================

    struct ParameterData{
        uint32_t type{moxGraphTypeUndefined};
        struct{
            ImageType type;
        } imageSpecification;
        std::string name{"undefined"};
        void* data{nullptr};
    };

    class GraphRender2 final{
    public:
        
        GraphRender2() = default;
        GraphRender2(GraphRender2& input) = delete;
        GraphRender2(const GraphRender2& input) = delete;
        GraphRender2& operator=(GraphRender2& input) = delete;
        GraphRender2& operator=(const GraphRender2& input) = delete;

        GraphRender2(GraphRender2&& input) noexcept {
            processCopy(std::move(input));
        }
        GraphRender2& operator=(GraphRender2&& input) noexcept{
            processCopy(std::move(input));
            return *this;
        }

        ~GraphRender2()noexcept{
            processDelete();
        }

        [[nodiscard]] ErrorDataOutput<void> execute(VkCommandBuffer &cmd , const std::string name) noexcept{
            return processExecution(cmd , name);
        }

        [[nodiscard]] ErrorDataOutput<void> processInputParameter(FrameTargets* data) noexcept{

            if(!data){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt add the FrameTargets as an input parameter , since its nullptr")});
            }

            // position
            {   
                {
                    const auto result = processInputParameter(data->positionImage[0].get() , "positionImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->positionImage[1].get() , "positionImage_1");
                    if(!result.has_value()) return result;
                }
            }

            // depth
            {
                {
                    const auto result = processInputParameter(data->depthImage[0].get() , "depthImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->depthImage[1].get() , "depthImage_1");
                    if(!result.has_value()) return result;
                }
            }

            // normal
            {
                {
                    const auto result = processInputParameter(data->normalImage[0].get() , "normalImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->normalImage[1].get() , "normalImage_1");
                    if(!result.has_value()) return result;
                }
            }

            // HDR image
            {
                {
                    const auto result = processInputParameter(data->HDRImage[0].get() , "HDRImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->HDRImage[1].get() , "HDRImage_1");
                    if(!result.has_value()) return result;
                }
            }

            // indirect
            {
                {
                    const auto result = processInputParameter(data->indirectImage[0].get() , "indirectImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->indirectImage[1].get() , "indirectImage_1");
                    if(!result.has_value()) return result;
                }
            }

            // diffuse
            {
                {
                    const auto result = processInputParameter(data->diffuseImage[0].get() , "diffuseImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->diffuseImage[1].get() , "diffuseImage_1");
                    if(!result.has_value()) return result;
                }
            }

            // specular
            {
                {
                    const auto result = processInputParameter(data->specularImage[0].get() , "specularImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->specularImage[1].get() , "specularImage_1");
                    if(!result.has_value()) return result;
                }
            }

            // shadows
            {
                {
                    const auto result = processInputParameter(data->shadowImage[0].get() , "shadowImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->shadowImage[1].get() , "shadowImage_1");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = processInputParameter(data->filteredShadowImage.get() , "filteredShadowImage");
                    if(!result.has_value()) return result;
                }
            }

            // ambient occlusion
            {
                {
                    const auto result = processInputParameter(data->ambientOcclusionImage[0].get() , "ambientOcclusionImage_0");
                    if(!result.has_value()) return result;
                }
                
                {
                    const auto result = processInputParameter(data->ambientOcclusionImage[1].get() , "ambientOcclusionImage_1");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = processInputParameter(data->filteredAmbientOcclusion.get() , "filteredAmbientOcclusion");
                    if(!result.has_value()) return result;
                }
            }

            // albedo / roughness / metallic
            {
                {
                    const auto result = processInputParameter(data->albedoImage.get() , "albedoImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = processInputParameter(data->roughnessImage.get() , "roughnessImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = processInputParameter(data->metallicImage.get() , "metallicImage");
                    if(!result.has_value()) return result;
                }
            }


            // ambient
            {
                {
                    const auto result = processInputParameter(data->ambientImage.get() , "ambientImage");
                    if(!result.has_value()) return result;
                }
            }


            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> processInputParameter(RenderImage2* image , const std::string name) noexcept{
            if(!image){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt add the image as an input parameter , since its nullptr")});
            }
            if(name.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL ,"failed to add image , since the name for that image is nullptr"});
            }

            if(allAttachments.contains(name)) return {};

            ParameterData data{};
            data.imageSpecification.type = ImageType::RenderImage2;
            data.type = moxGraphTypeImage;
            data.name = name;
            data.data = image;

            allAttachments.insert({name,  data});

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> processInputParameter(Skybox2* image , const std::string name) noexcept{
            if(!image){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt add the image as an input parameter , since its nullptr")});
            }
            if(name.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL ,"failed to add image , since the name for that image is nullptr"});
            }

            if(allAttachments.contains(name)) return {};

            ParameterData data{};
            data.imageSpecification.type = ImageType::Skybox2;
            data.type = moxGraphTypeImage;
            data.name = name;
            data.data = image;

            allAttachments.insert({name,  data});

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> processInputParameter(Texture2* image , const std::string name) noexcept{
            if(!image){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt add the image as an input parameter , since its nullptr")});
            }
            if(name.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL ,"failed to add image , since the name for that image is nullptr"});
            }

            if(allAttachments.contains(name)) return {};

            ParameterData data{};
            data.imageSpecification.type = ImageType::Texture2;
            data.type = moxGraphTypeImage;
            data.name = name;
            data.data = image;

            allAttachments.insert({name,  data});

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> processInputParameter(BufferBase2* buffer , const std::string name) noexcept{

            if(!buffer){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt add the buffer as an input parameter , since its nullptr")});
            }
            if(name.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL ,"failed to add buffer , since the name for that buffer is nullptr"});
            }

            if(allAttachments.contains(name)) return {};

            ParameterData data{};
            data.type = moxGraphTypeBuffer;
            data.name = name;
            data.data = buffer;

            allAttachments.insert({name,  data});

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> addGraphPass2(GraphPass2* pass , const std::string pathName) noexcept{
            return processInputGraphPass(pathName , pass);
        }

    private:

        void processCopy(GraphRender2&& input) noexcept{

            // the old check compared the content of the two maps : two empty graphs are equal , and the move did nothing
            if(this == &input) return;
            processDelete();
            allAttachments = std::move(input.allAttachments);
            graphPasses = std::move(input.graphPasses);
        }

        void processDelete() noexcept{
            allAttachments.clear();
            graphPasses.clear();
        }

        [[nodiscard]] ErrorDataOutput<void> processInputGraphPass(const std::string name , GraphPass2* pass) noexcept {

            const auto reqs = pass->acquireRequirements();
            std::vector<GraphParameterSource> sources{};

            for(auto& r : reqs){

                const auto it = allAttachments.find(r.name);
                if(it == allAttachments.end()){
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt add graph pass to render , since the attachment [[  {}  ]] , wasnt founded" , r.name)});
                }

                const ParameterData data = it->second;
                if(data.type != r.type) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt add graph pass to render , since the attachment [[  {}  ]] , has different types" , r.name)});
                if(r.type == moxGraphTypeImage){
                    if(r.imageSpecification.type != data.imageSpecification.type) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt add graph pass to render , since the attachment [[  {}  ]] , image types are different from each other" , r.name)});
                }
                GraphParameterSource source{};
                source.name = data.name;
                source.location=  data.data;

                sources.push_back(source);
            }
            const auto result = pass->postProcessGraphRenderLocations(sources);
            if(!result.has_value()) return result;

            if(graphPasses.contains(name)){
                graphPasses[name].push_back(pass);
            }else{
                graphPasses.insert({name , {pass}});
            }

            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> processExecution(VkCommandBuffer &cmd,  const std::string name){

            if(!graphPasses.contains(name)) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to execute Render -> {} , since it doesnt exist in graph render" , name)});

            // operator[] made an empty graph out of a name that was not there : a typo in the name was a success that did nothing
            const auto found = graphPasses.find(name);
            if(found == graphPasses.end()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to execute the graph -> {} , since there is no graph with this name" , name)});
            }

            for(auto& graph : found->second){

                if(!graph->requestExecution()) continue;

                {
                    const auto result = graph->translateBarriers(cmd);
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = graph->executePass(cmd);
                    if(!result.has_value()) return result;
                }
            }

            return {};
        }


        std::unordered_map<std::string , ParameterData> allAttachments{};
        std::unordered_map<std::string , std::vector<GraphPass2*>> graphPasses;
    };
}