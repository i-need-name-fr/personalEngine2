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

            file << std::format("{}" , message);
            file.close();
        } 
        
        void printErrorType(const ErrorSignature type , const std::string message, std::source_location location) noexcept{

            logAction("=========================== MESSAGE ===========================\n||\n||\n");

            logAction("--------------------------- ERROR_TYPE ---------------------------\n||\n||\n");

            switch(type){
                case mox::ErrorSignature::ErrorTypeWARNING : {
                    logAction("|| {{    ERROR_TYPE    }} : [[   WARNING   ]]\n||");
                    break;
                }
                case mox::ErrorSignature::ErrorTypeMEDIUM : {
                    logAction("|| {{    ERROR_TYPE    }} : [[   MEDIUM   ]]\n||");

                    break;
                }
                case mox::ErrorSignature::ErrorTypeCRITICAL : {
                    logAction("|| {{    ERROR_TYPE    }} : [[   CRITICAL   ]]\n||");

                    break;
                }
                case mox::ErrorSignature::ErrorTypeFAILED : {
                    logAction("|| {{    ERROR_TYPE    }} : [[   FAILED   ]]\n||");

                    break;
                }
                case mox::ErrorSignature::ErrorTypeFAILED_CREATION : {
                    logAction("|| {{    ERROR_TYPE    }} : [[   FAILED_CREATION   ]]\n||");

                    break;
                }
                case mox::ErrorSignature::ErrorTypeSUCCESS : {
                    logAction("|| {{    ERROR_TYPE    }} : [[[      SUCCESS     ]]]\n||");

                    break;
                }
                default: {
                    logAction("|| {{    ERROR_TYPE    }} : [[   UNDEFINED_ERROR   ]]\n||");
                    break;
                }
            }

            logAction("\n||\n------------------------- SOURCE LOCATION -------------------------\n||\n");

            logAction(std::format("|| {{    FILE_LOCATION   }} : {}\n||\n" , location.file_name()));
            logAction(std::format("|| {{    FUNCTION_NAME   }} : {}\n||\n" , location.function_name()));
            logAction(std::format("|| {{    LINE    }} : {}\n||\n" , location.line()));

            logAction("--------------------------- INFORMATION ---------------------------\n||\n");

            logAction(std::format("|| {{    Message    }} : {}\n||\n" , message));

            logAction("=========================== END MESSAGE ===========================\n\n\n\n\n");

        }

        [[gnu::hot]] inline void setupLogMessage(const mox::ErrorSignature errorType , const std::string message ,const std::string fileOutput = LOG_FILE_OUTPUT,  std::source_location location = std::source_location::current() ){
            printErrorType(errorType , message , location);
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