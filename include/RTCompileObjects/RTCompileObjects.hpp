#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>

namespace mox{


/*
#ifdef SPIRV_COMPILATION_REQUIRED

#else

#endif
*/

    // name should be without data type
    inline std::string getShaderPath(const std::string path) noexcept{
        #ifdef SPIRV_COMPILATION_REQUIRED
            return path + ".hlsl";
        #else           
            return path + ".spv";
        #endif
    }


    class RTCompileObjects {
    public:

        RTCompileObjects() = default;

    protected:

        inline static std::optional<bool> changedShaderStructure;

        static bool checkValidShaderStructure() noexcept{
            if(changedShaderStructure.has_value()) return changedShaderStructure.value();

            std::error_code error;
            if(!std::filesystem::exists(stampPath() , error)) return true;

            std::ifstream input(structurePath() , std::ios::binary);
            std::ifstream stamp(stampPath() , std::ios::binary);

            const bool status = std::equal(
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>(),
                std::istreambuf_iterator<char>(stamp)
            );
            changedShaderStructure = status;
            return status;
        }

        static void commitShaderStructureStamp() noexcept{
            if(changedShaderStructure.has_value() && changedShaderStructure.value()) return;

            std::error_code error;
            std::filesystem::create_directories(std::filesystem::path(stampPath()).parent_path() , error);
            std::filesystem::copy_file(structurePath() , stampPath() , std::filesystem::copy_options::overwrite_existing , error);
            if(error){
                std::cerr << "failed to reupdate the shaderStructure : " << error.message() << '\n';
            }
        }

    private:

        static const char* structurePath() noexcept{ return "../include/shaderStructures/shaderStructure.h"; }
        static const char* stampPath() noexcept{ return "../ShaderModuleCache/TempShaderStructure.h"; }
    };
}
