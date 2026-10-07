#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <fstream>
#include <source_location>
#include <format>
#include <bit>
#include <expected>

#define LOG_FILE_OUTPUT "../log.txt"

namespace mox{

    enum class ErrorSignature{
        ErrorTypeCRITICAL = 0lu,
        ErrorTypeWARNING = 1lu,
        ErrorTypeSUCCESS = 2lu,
        ErrorTypeFAILED_CREATION = 3lu,
        ErrorTypeMEDIUM = 4lu,
        ErrorTypeFAILED = 5lu,
        ErrorTypeNONE = 3454lu
    };

    inline static constexpr uint32_t totalMessageCount = 10ul;

    const ErrorSignature MOX_ERROR_TYPE_CRITICAL = ErrorSignature::ErrorTypeCRITICAL;
    const ErrorSignature MOX_ERROR_TYPE_FAILED = ErrorSignature::ErrorTypeFAILED;
    const ErrorSignature MOX_ERROR_TYPE_NONE = ErrorSignature::ErrorTypeNONE;
    const ErrorSignature MOX_ERROR_TYPE_MEDIUM = ErrorSignature::ErrorTypeMEDIUM;
    const ErrorSignature MOX_ERROR_TYPE_FAILED_CREATION = ErrorSignature::ErrorTypeFAILED_CREATION;
    const ErrorSignature MOX_ERROR_TYPE_WARNING = ErrorSignature::ErrorTypeWARNING;
    const ErrorSignature MOX_ERROR_TYPE_SUCCESS = ErrorSignature::ErrorTypeSUCCESS;


    using ErrorDataType = std::pair<ErrorSignature , std::string>;
    template<typename T>
    using ErrorDataOutput = std::expected<T , ErrorDataType>;

    struct EngineLogger_T{
    private:
        [[gnu::hot]] static inline void logAction(const std::string& message ,const std::string fileOutput = LOG_FILE_OUTPUT) noexcept{
            const bool nullMessage = message.empty();
            std::ofstream file(fileOutput , std::ios::app);
            if(!file.is_open()){
                std::cerr << std::format("WARNING : couldnt find the file for the logging. \nGiven path is {}" , fileOutput) << '\n';
                return;
            }

            file << std::format("LOG MESSAGE : {}" , message);
            file.close();
        } 
        

        [[gnu::hot]] static inline void setupLogMessage(const mox::ErrorSignature errorType , const std::string message ,const std::string fileOutput = LOG_FILE_OUTPUT,  std::source_location location = std::source_location::current() ){
            std::string error;
            switch(errorType){
                case mox::ErrorSignature::ErrorTypeWARNING : {
                    error = std::format("\n----------------------\nERROR_TYPE_WARNING :\n LOCATION -> FUCTION |{}| ,\n FILE |{}| ,\nLINE |{}|\n====================\nMESSAGE : {}\n====================\n\n" , location.function_name() , location.file_name() , location.line() , message);
                    break;
                }
                case mox::ErrorSignature::ErrorTypeMEDIUM : {
                    error = std::format("\n----------------------\nERROR_TYPE_MEDIUM :\n LOCATION -> FUCTION |{}| ,\n FILE |{}| ,\nLINE |{}|\n====================\nMESSAGE : {}\n====================\n\n" , location.function_name() , location.file_name() , location.line() , message);

                    break;
                }
                case mox::ErrorSignature::ErrorTypeCRITICAL : {
                    error = std::format("\n----------------------\nERROR_TYPE_CRITICAL :\n LOCATION -> FUCTION |{}| ,\n FILE |{}| ,\nLINE |{}|\n====================\nMESSAGE : {}\n====================\n\n" , location.function_name() , location.file_name() , location.line() , message);

                    break;
                }
                case mox::ErrorSignature::ErrorTypeFAILED : {
                    error = std::format("\n----------------------\nERROR_TYPE_FAILED :\n LOCATION -> FUCTION |{}| ,\n FILE |{}| ,\nLINE |{}|\n====================\nMESSAGE : {}\n====================\n\n" , location.function_name() , location.file_name() , location.line() , message);
                    break;
                }
                case mox::ErrorSignature::ErrorTypeFAILED_CREATION : {
                    error = std::format("\n----------------------\nERROR_TYPE_FAILED_CREATION : \nLOCATION -> FUCTION |{}| ,\n FILE |{}| ,\nLINE |{}|\n====================\nMESSAGE : {}\n====================\n\n" , location.function_name() , location.file_name() , location.line() , message);
                    break;
                }
                case mox::ErrorSignature::ErrorTypeSUCCESS : {
                    error = std::format("\n----------------------\nERROR NONE : SUCCESS :\n====================\nMESSAGE -> {}\n====================\n\n" , message);
                    break;
                }
                default: {
                    error = std::format("\n----------------------\nERROR_TYPE_NONE :\n LOCATION -> FUCTION |{}| ,\n FILE |{}| ,\nLINE |{}|\n====================\nMESSAGE : {}\n====================\n\n" , location.function_name() , location.file_name() , location.line() , message);
                    break;
                }
            }

            logAction(error , fileOutput);
        }
    public:
        
        EngineLogger_T(const std::string path = LOG_FILE_OUTPUT){
            std::ofstream cleanFile(path , std::ios::out);
            if(!cleanFile.is_open()) return;
            cleanFile.close();
        }

        struct messageCreateInfo{
            mox::ErrorSignature errorType;
            std::string message;
        };

        void operator()(const void* info , const std::string &path = LOG_FILE_OUTPUT , std::source_location location = std::source_location::current()){
            if(!info) return;
            messageCreateInfo* data = std::bit_cast<messageCreateInfo*>(info);
            setupLogMessage(data->errorType , data->message , path , location);
        }

        void operator()(const ErrorDataType *info, const std::string &path = LOG_FILE_OUTPUT , std::source_location location = std::source_location::current()){
            if(!info) return;
            setupLogMessage(info->first , info->second , path , location);
        }

        void operator()(const ErrorDataType &info, const std::string &path = LOG_FILE_OUTPUT , std::source_location location = std::source_location::current()){
            setupLogMessage(info.first , info.second , path , location);
        }

    };

    inline EngineLogger_T engineLogger{};

}

#endif