
file(GLOB IMGUI_DATA 
        ${CMAKE_SOURCE_DIR}/imgui/*.cpp
        ${CMAKE_SOURCE_DIR}/imgui/backends/*.cpp)

add_library(imgui STATIC ${IMGUI_DATA})

target_include_directories(imgui PUBLIC ${CMAKE_SOURCE_DIR}/imgui
                            ${CMAKE_SOURCE_DIR}/imgui/backends)

target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)
