#pragma once

#include "Instance.hpp"

namespace mox{

    class Window{
        ErrorDataOutput<void> init(void* inputData , GLFWcursorposfun mouse , GLFWkeyfun key ){
            if(!glfwInit()) throw std::runtime_error("lol");
            glfwWindowHint(GLFW_CLIENT_API , GLFW_NO_API);
            glfwWindowHint(GLFW_MAXIMIZED , GLFW_TRUE);
            glfwWindowHint(GLFW_RESIZABLE , GLFW_FALSE);

            window = glfwCreateWindow(2560 , 1440 , "window" , nullptr , nullptr);
            if(!window){
                ErrorDataType error = {MOX_ERROR_TYPE_CRITICAL , "failed to create the window"};
                return std::unexpected(error);
            }
            glfwSetInputMode(window , GLFW_CURSOR , GLFW_CURSOR_DISABLED);
            // без этого mousePosCallBack разыменовывает nullptr
            glfwSetWindowUserPointer(window , inputData);
            glfwSetCursorPosCallback(window , mouse);
            glfwSetKeyCallback(window , key);
            return {};
        }

    public:

        GLFWwindow* window = nullptr;

        struct WindowCreateInfo{
            GLFWcursorposfun mouse; 
            GLFWkeyfun key;
            void* pUser;
        };

        Window(const WindowCreateInfo& info){
            if(!info.mouse || !info.key) throw std::runtime_error("functions for key and mouse call backs are nullptr");
            std::cout << "creating the window!\n";
            auto result = init(info.pUser , info.mouse , info.key);
            if(result.has_value()){

                ErrorDataType res = {MOX_ERROR_TYPE_SUCCESS , "window has been created successfully"};
                engineLogger(&res);

            }else{
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }
        Window() = delete;
        Window(Window& input) = delete;
        Window& operator=(Window& input) = delete;
        Window(Window&& input){
            window = std::exchange(input.window , nullptr);
        }
        Window& operator=(Window&& input){
            window = std::exchange(input.window , nullptr);
            return *this;
        }

        ~Window(){
            if(window){
                glfwDestroyWindow(window);
                glfwTerminate();
                ErrorDataType res = {MOX_ERROR_TYPE_NONE , "window has been destroyed"};
                engineLogger(&res);
            }
        }
    };
}