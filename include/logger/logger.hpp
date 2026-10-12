#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <fstream>
#include <source_location>
#include <format>
#include <bit>
#include <expected>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>
#include <chrono>
#include <ctime>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <unordered_map>

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

    // how many times a message that can repeat every frame is written ( the callers count their messages with it )
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

    // A message is a page : the number , the time , the type , the place and the text . the number is in the first and in the last line ,
    // so it is easy to see where a message starts and where it ends , and to find it ( "MESSAGE #0043" ) .
    //
    //   ================================ MESSAGE #0043 ================================
    //   ||
    //   || [[   TIMER : 20:42:14.229   |   +03.230 s   |   FRAME : 1204   |   THREAD : 1   ]]
    //   ||
    //   ----------------------------------- TYPE -----------------------------------
    //   || [[   CRITICAL   ]]
    //   ||
    //   ----------------------------------- WHERE ----------------------------------
    //   || FILE     : deferredRender.hpp
    //   || LINE     : 288
    //   || FUNCTION : mox::DeferredRender::executePass
    //   ||
    //   ----------------------------------- WHAT -----------------------------------
    //   || the text of the message ( the lines are wrapped , the new lines of the text stay )
    //   ||
    //   ============================== END MESSAGE #0043 ==============================
    //
    // the whole page is built first and written with ONE write under a mutex , so the pages of different threads never mix
    struct EngineLogger_T{
    private:

        inline static constexpr size_t pageWidth = 80;
        inline static constexpr size_t textWidth = 66;

        std::mutex mutex{};
        std::ofstream defaultFile{};
        // every file has its own numbers : the pages of another file do not leave a gap in the numbers of the log
        std::unordered_map<std::string , uint32_t> messageIds{};
        std::atomic<uint32_t> frame{UINT32_MAX};
        const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

        // a line like "====== TITLE ======" of the width of the page
        [[nodiscard]] static std::string banner(const std::string& title , const char fill){
            const std::string middle = title.empty() ? std::string{} : (" " + title + " ");
            const size_t total = middle.size() >= pageWidth ? 0 : pageWidth - middle.size();
            const size_t left = total / 2;
            return std::string(left , fill) + middle + std::string(total - left , fill) + "\n";
        }

        [[nodiscard]] static const char* typeName(const ErrorSignature type) noexcept{
            switch(type){
                case ErrorSignature::ErrorTypeWARNING : return "WARNING";
                case ErrorSignature::ErrorTypeMEDIUM : return "MEDIUM";
                case ErrorSignature::ErrorTypeCRITICAL : return "CRITICAL";
                case ErrorSignature::ErrorTypeFAILED : return "FAILED";
                case ErrorSignature::ErrorTypeFAILED_CREATION : return "FAILED_CREATION";
                case ErrorSignature::ErrorTypeSUCCESS : return "SUCCESS";
                default : return "UNDEFINED_ERROR";
            }
        }

        // the name of the file without the folders ( the path of an include is long : .../include/Buffer2/../Buffers/../x.hpp )
        [[nodiscard]] static std::string_view fileName(const std::string_view path) noexcept{
            const size_t slash = path.find_last_of("/\\");
            return slash == std::string_view::npos ? path : path.substr(slash + 1);
        }

        // the text in lines of the width of the page : the new lines of the message stay , the words are not cut
        [[nodiscard]] static std::vector<std::string> wrap(const std::string& text){
            std::vector<std::string> lines;
            std::istringstream paragraphs(text);
            std::string paragraph;
            while(std::getline(paragraphs , paragraph)){
                if(!paragraph.empty() && paragraph.back() == '\r') paragraph.pop_back();
                std::istringstream words(paragraph);
                std::string word , line;
                while(words >> word){
                    if(!line.empty() && line.size() + 1 + word.size() > textWidth){
                        lines.push_back(line);
                        line.clear();
                    }
                    line += (line.empty() ? "" : " ") + word;
                }
                lines.push_back(line);
            }
            if(lines.empty()) lines.push_back("");
            return lines;
        }

        [[nodiscard]] static std::string wallClock() noexcept{
            const auto now = std::chrono::system_clock::now();
            const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
            const int millis = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);

            std::tm local{};
#if defined(_WIN32)
            localtime_s(&local , &seconds);
#else
            localtime_r(&seconds , &local);
#endif
            return std::format("{:02}:{:02}:{:02}.{:03}" , local.tm_hour , local.tm_min , local.tm_sec , millis);
        }

        // 1 for the first thread that wrote a message , 2 for the next one and so on : easier to read than the id of the thread
        [[nodiscard]] static uint32_t threadNumber() noexcept{
            static std::atomic<uint32_t> counter{0};
            thread_local const uint32_t number = ++counter;
            return number;
        }

        // the page of one message ( the number is given by the caller : it is taken under the mutex , so the numbers go in the order of the file )
        [[nodiscard]] std::string buildPage(const uint32_t id , const ErrorSignature type , const std::string& message , const std::source_location& location) const{
            const double sinceStart = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
            const uint32_t currentFrame = frame.load(std::memory_order_relaxed);
            const std::string frameText = currentFrame == UINT32_MAX ? std::string("-") : std::to_string(currentFrame);

            std::string page;
            page.reserve(512 + message.size() * 2);

            page += banner(std::format("MESSAGE #{:04}" , id) , '=');
            page += "||\n";
            page += std::format("|| [[   TIMER : {}   |   +{:06.3f} s   |   FRAME : {}   |   THREAD : {}   ]]\n" , wallClock() , sinceStart , frameText , threadNumber());
            page += "||\n";

            page += banner("TYPE" , '-');
            page += std::format("|| [[   {}   ]]\n" , typeName(type));
            page += "||\n";

            page += banner("WHERE" , '-');
            page += std::format("|| FILE     : {}\n" , fileName(location.file_name()));
            page += std::format("|| LINE     : {}\n" , location.line());
            page += std::format("|| FUNCTION : {}\n" , location.function_name());
            page += "||\n";

            page += banner("WHAT" , '-');
            for(const auto& line : wrap(message)){
                page += "|| ";
                page += line;
                page += '\n';
            }
            page += "||\n";

            page += banner(std::format("END MESSAGE #{:04}" , id) , '=');
            page += "\n\n";
            return page;
        }

        [[gnu::hot]] void setupLogMessage(const mox::ErrorSignature errorType , const std::string& message , const std::string& fileOutput = LOG_FILE_OUTPUT , std::source_location location = std::source_location::current()){
            // one lock for the number , the page and the write : the pages are in the file in the order of their numbers
            std::lock_guard<std::mutex> lock(mutex);

            const std::string page = buildPage(++messageIds[fileOutput] , errorType , message , location);

            if(fileOutput == LOG_FILE_OUTPUT){
                if(!defaultFile.is_open()){
                    std::cerr << std::format("WARNING : couldnt find the file for the logging. \nGiven path is {}\n" , fileOutput);
                    return;
                }
                // flush : the page is in the file even if the program crashes right after it
                defaultFile.write(page.data() , static_cast<std::streamsize>(page.size()));
                defaultFile.flush();
                return;
            }

            // another file was asked for
            std::ofstream file(fileOutput , std::ios::app);
            if(!file.is_open()){
                std::cerr << std::format("WARNING : couldnt find the file for the logging. \nGiven path is {}\n" , fileOutput);
                return;
            }
            file.write(page.data() , static_cast<std::streamsize>(page.size()));
        }

    public:

        EngineLogger_T(const std::string path = LOG_FILE_OUTPUT){
            // a new run starts a new log
            defaultFile.open(path , std::ios::out | std::ios::trunc);
        }

        EngineLogger_T(const EngineLogger_T&) = delete;
        EngineLogger_T& operator=(const EngineLogger_T&) = delete;

        struct messageCreateInfo{
            mox::ErrorSignature errorType;
            std::string message;
        };

        // the number of the current frame ( it is written in every message ) : the frame loop gives it , until then the messages say "-"
        void setFrame(const uint32_t frameIndex) noexcept{
            frame.store(frameIndex , std::memory_order_relaxed);
        }

        void operator()(const void* info , const std::string &path = LOG_FILE_OUTPUT , std::source_location location = std::source_location::current()){
            if(!info) return;
            const messageCreateInfo* data = static_cast<const messageCreateInfo*>(info);
            setupLogMessage(data->errorType , data->message , path , location);
        }

        void operator()(const ErrorDataType *info, const std::string &path = LOG_FILE_OUTPUT , std::source_location location = std::source_location::current()){
            if(!info) return;
            setupLogMessage(info->first , info->second , path , location);
        }

        void operator()(const ErrorDataType &info, const std::string &path = LOG_FILE_OUTPUT , std::source_location location = std::source_location::current()){
            setupLogMessage(info.first , info.second , path , location);
        }

        void operator()(const ErrorDataOutput<void> &info, const std::string &path = LOG_FILE_OUTPUT , std::source_location location = std::source_location::current()){
            if(info.has_value()) return;
            setupLogMessage(info.error().first , info.error().second , path , location);
        }
    };

    inline EngineLogger_T engineLogger{};

}

#endif
