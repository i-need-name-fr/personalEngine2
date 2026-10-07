#include <iostream>
#include "include/Buffer2/GeneralBuffer.hpp"


int main(){
    std::cout << "hello world\n";
    try{

        mox::VulkanContext::CreateInfo info{};
        info.asyncMultiThreadQueueCount = 4;
        info.multithreadQueueCount = 1;

        mox::VulkanContext context(info);

        std::vector<uint32_t> inputData{3974 , 8273, 1821, 982392 , 121, 129819, 283, 111 , 0, 4545};

        mox::GeneralBuffer2<mox::BufferType::Device , true>::CreateInfo data{};
        data.context = &context;
        data.size = sizeof(uint32_t) * 10;
        data.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;


        mox::GeneralBuffer2<mox::BufferType::Device ,true> buffer(data , mox::translateToSpan<uint32_t>(inputData));
    }
    catch(std::exception &e){
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}