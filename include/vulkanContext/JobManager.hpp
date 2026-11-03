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
#include <mutex>
#include <shared_mutex>
#include <deque>
#include <vector>
#include <string>
#include <memory>
#include "Instance.hpp"

namespace mox
{

    struct FunctionParameter
    {
        std::atomic<uint32_t> status;
    };


    using moxJobManagerFunctionType = std::function<uint32_t()>;

    struct FunctionSource
    {
        moxJobManagerFunctionType function;
        std::string name{};
        std::shared_ptr<FunctionParameter> parameter{};
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

        struct CreateInfo
        {

        };

        explicit JobManager(const CreateInfo& info)
        {
            state = std::make_shared<State>();
            for(uint32_t i = 0 ; i < MOX_JOB_MANAGER_THREAD_COUNT ; i++)
                std::thread(&JobManager::_implProcessThread , state).detach();
        }

        ~JobManager() noexcept
        {
            processDelete();
        }

        void waitFor(const std::string name) noexcept
        {
            _implWaitForFunc(name);
        }

        uint32_t getStatus(const std::string name) noexcept
        {
            return _implGetFunctionStatus(name);
        }

        void addJob(const std::string name ,const moxJobManagerFunctionType job) noexcept
        {
            _implAddJob(job , name);
        }

        void waitForAll() noexcept
        {
            if(!state) return;
            std::vector<std::shared_ptr<FunctionParameter>> parameters{};
            {
                std::shared_lock lock(state->m);
                parameters.reserve(state->functionStatuses.size());
                for(const auto& [name , data] : state->functionStatuses)
                    parameters.push_back(data);
            }
            for(const auto& parameter : parameters)
            {
                uint32_t current = parameter->status.load(std::memory_order::acquire);
                while(current != moxFunctionFinished && current != moxFunctionFinishedWithError)
                {
                    parameter->status.wait(current);
                    current = parameter->status.load(std::memory_order::acquire);
                }
            }
        }

    private:

        inline static constexpr uint32_t MOX_JOB_MANAGER_THREAD_COUNT = 10;

        struct State
        {
            mutable std::shared_mutex m;
            std::condition_variable_any var{};
            std::unordered_map<std::string , std::shared_ptr<FunctionParameter>> functionStatuses;
            std::deque<FunctionSource> jobs{};
            std::stop_source stop{};
        };

        void processCopy(JobManager&& input) noexcept
        {
            processDelete();
            state = std::move(input.state);
        }

        void processDelete() noexcept
        {
            if(state) state->stop.request_stop();
            state.reset();
        }

        static void _implProcessThread(std::shared_ptr<State> s) noexcept
        {
            const std::stop_token tk = s->stop.get_token();
            while(true)
            {
                FunctionSource job{};

                {
                    std::unique_lock guard(s->m);
                    if(!s->var.wait(guard , tk , [&](){return !s->jobs.empty();})) return;

                    job = std::move(s->jobs.front());
                    s->jobs.pop_front();
                }

                if(job.function)
                {
                    job.parameter->status.store(moxFunctionNotFinished , std::memory_order::release);
                    uint32_t result = moxFunctionFinishedWithError;
                    try
                    {
                        result = job.function();
                    }
                    catch(...)
                    {
                        result = moxFunctionFinishedWithError;
                    }
                    job.parameter->status.store(result , std::memory_order::release);
                    job.parameter->status.notify_all();
                }
            }
        }

        void _implWaitForFunc(const std::string& name) const noexcept
        {
            std::shared_ptr<FunctionParameter> parameter{};
            {
                if(!state) return;
                std::shared_lock lock(state->m);                   // only to find the element
                const auto it = state->functionStatuses.find(name);
                if(it == state->functionStatuses.end()) return;
                parameter = it->second;                            // the status cannot die while we wait
            }
            uint32_t current = parameter->status.load(std::memory_order::acquire);
            while(current != moxFunctionFinished && current != moxFunctionFinishedWithError)
            {
                parameter->status.wait(current);                   // sleeps while the value == current
                current = parameter->status.load(std::memory_order::acquire);
            }
        }

        [[nodiscard]] uint32_t _implGetFunctionStatus(const std::string& name ) noexcept
        {
            std::shared_ptr<FunctionParameter> parameter{};
            {
                if(!state) return moxFunctionNotExists;
                std::shared_lock lock(state->m);
                const auto it = state->functionStatuses.find(name);
                if(it == state->functionStatuses.end()) return moxFunctionNotExists;
                parameter = it->second;
            }

            const uint32_t status = parameter->status.load(std::memory_order::acquire);
            if(status == moxFunctionFinished || status == moxFunctionFinishedWithError)
            {
                std::unique_lock lock(state->m);
                const auto it = state->functionStatuses.find(name);
                // maybe the same name was added again in the meantime : erase only the one we have read
                if(it != state->functionStatuses.end() && it->second == parameter) state->functionStatuses.erase(it);
            }
            return status;
        }

        void _implAddJob(const moxJobManagerFunctionType function , const std::string name) noexcept
        {
            auto parameter = std::make_shared<FunctionParameter>();
            parameter->status.store(moxFunctionNotStarted , std::memory_order::relaxed);
            if(!state) return;
            {
                std::unique_lock guard(state->m);
                state->functionStatuses.insert_or_assign(name , parameter);
                state->jobs.push_back({function , name , parameter});
            }
            state->var.notify_one();
        }

        std::shared_ptr<State> state{};
    };

    using pJobManager = std::shared_ptr<JobManager>;
}


