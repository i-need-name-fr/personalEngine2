#pragma once

// the shader classes that the passes ( generalPipelines2 ) use are chosen in ONE place by SPIRV_COMPILATION_REQUIRED :
//   defined     - ShaderPipe2 / ShaderObject2 / ShaderModule2 : the passes get .hlsl files and compile them at run time when they changed
//   not defined - ShaderPipe / ShaderObject / ShaderModule   : the passes get the compiled .spv files
// a pass uses ShaderPipeType / pShaderPipeType / ShaderObjectType / ShaderModuleType , and the paths come from getShaderPath ( RTCompileObjects.hpp ) , which
// gives the file with the right extension for the same condition
#ifdef SPIRV_COMPILATION_REQUIRED
    #include "ShaderPipe2.hpp"
    #include "../ShaderModule2/ShaderModule2.hpp"
#else
    #include "../ShaderPipe/ShaderPipe.hpp"
    #include "../ShaderModule/ShaderModule.hpp"
#endif
#include "../RTCompileObjects/RTCompileObjects.hpp"

namespace mox{

#ifdef SPIRV_COMPILATION_REQUIRED
    using ShaderPipeType = ShaderPipe2;
    using pShaderPipeType = pShaderPipe2;
    using ShaderObjectType = ShaderObject2;
    using ShaderModuleType = ShaderModule2;
#else
    using ShaderPipeType = ShaderPipe;
    using pShaderPipeType = pShaderPipe;
    using ShaderObjectType = ShaderObject;
    using ShaderModuleType = ShaderModule;
#endif

}
