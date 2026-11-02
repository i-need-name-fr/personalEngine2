#pragma once

#include <iostream>
#include <thread>
#include <atomic>
#include <stdint.h>
#include <unordered_map>
#include <ranges>
#include <array>
#include <functional>
#include <condition_variable>
#include "vulkanBase.hpp"

namespace mox
{

    struct FunctionParameter
    {
        std::atomic<uint32_t> status;
    };


    using moxJobManagerFunctionType = std::function<void(std::atomic<uint32_t>)>;

    struct FunctionSource
    {
        moxJobManagerFunctionType function;
        std::string name{};
    };


    inline static constexpr uint32_t moxFunctionNotStarted = 2;
    inline static constexpr uint32_t moxFunctionNotFinished = 1;
    inline static constexpr uint32_t moxFunctionFinished = 0;
    inline static constexpr uint32_t moxFunctionNotExists = 3;
    inline static constexpr uint32_t moxFunctionFinishedWithError = 4;

    class JobManager final
    {
    public:

        SetTypicalOperators(JobManager);
        SetTypicalTransferOperators(JobManager);


    private:

        void processCopy(JobManager&& input) noexcept
        {

        }

        void processDelete() noexcept
        {

        }

        [[nodiscard]] uint32_t _implGetFunctionStatus(const std::string& name ) noexcept
        {
            if(!functionStatuses.contains(name))
            {
                return moxFunctionNotExists;
            }

            const uint32_t status = functionStatuses[name].status.load(std::memory_order::acquire);
            if(status == moxFunctionFinished)
            {
                std::lock_guard<std::shared_mutex> guard(m);
                functionStatuses.erase(name);
            }
            return status;
        }

        void _implAddJob(const moxJobManagerFunctionType function , const std::string name) noexcept
        {
            {
                std::lock_guard<std::shared_mutex> guard(m);
                jobs.push_back({function , name});
                functionStatuses.try_emplace(name , moxFunctionNotStarted);
                var.notify_one();
            }
        }

        

        std::shared_mutex m;
        std::condition_variable var{};
        std::unordered_map<std::string , FunctionParameter> functionStatuses;
        std::vector<FunctionSource> jobs{};
        std::array<std::jthread , 10> threads{};
    };
}
