#pragma once

#include "Window.hpp"

namespace mox{
    class Player{
    private:

        const glm::vec3 up = {0.0f , 1.0f , 0.0f};
        const float speed = 10.0f;
        const float rotation_speed = 0.05f;

    public:

        Player() = default;
        Player(Player& input) = default;
        Player(Player&& input )= default;
        Player& operator=(const Player& input)
        {
            if (this != &input)
            {
                position = input.position;
                direction = input.direction;
                lastX = input.lastX;
                lastY = input.lastY;
                yaw = input.yaw;
                pitch = input.pitch;
                openedMenu = input.openedMenu;
                playerMoved = input.playerMoved;
            }
            return *this;
        }

        Player& operator=(Player&& input) noexcept
        {
            if (this != &input)
            {
                position = input.position;
                direction = input.direction;
                lastX = input.lastX;
                lastY = input.lastY;
                yaw = input.yaw;
                pitch = input.pitch;
                openedMenu = input.openedMenu;
                playerMoved = input.playerMoved;
            }
            return *this;
        }



        glm::vec3 position = {0.0f , 0.0f , 5.0f};
        glm::vec3 direction = {0.0f , 0.0f , -1.0f};

        float lastX = 0.0f;
        float lastY = 0.0f;

        float yaw = -90.0f;
        float pitch = 0.0f;

        const uint32_t width = 2560;
        const uint32_t height = 1440;
        
        bool openedMenu = false;
        bool firstMouseSample = true;   // was a function-local static in processMouseInput
        bool playerMoved = true;
        bool PT_required = false;

        uint32_t currentMode = 0;
        // default , albedo , normal , depth , indirect light ( DEFAULT_MODE .. INDIRECT_MODE ) , the R key cycles through all of them
        const uint32_t totalModes = 5;

        glm::vec3 calcRight() const noexcept{
            return glm::normalize(glm::cross(glm::vec3(direction) , glm::vec3(0.0f , 1.0f , 0.0f)));
        }

        void checkPitch()noexcept{
            if(pitch > 89.0f) pitch = 89.0f;
            if(pitch < -89.0f) pitch = -89.0f;
        }

        void calcFront()noexcept{
            if(!playerMoved) return;
            checkPitch();
            glm::vec3 front = glm::vec3(0.0f);
            front.x = glm::cos(glm::radians(pitch)) * glm::cos(glm::radians(yaw));
            front.y = glm::sin(glm::radians(pitch));
            front.z = glm::cos(glm::radians(pitch)) * glm::sin(glm::radians(yaw));

            front = glm::normalize(front);
            direction = front;
        }

        /// moves player either straight or backwards
        void movePlayer_straight_or_not(bool front, const float dt){
            calcFront();
            glm::vec3 currentPosition = glm::vec3(position);
            const glm::vec3 currentFront = glm::vec3(direction);
            if(front){
                currentPosition += currentFront * dt * speed;
            }else{
                currentPosition -= currentFront * dt * speed;
            }
            position = currentPosition;
            // we will be setting the playerMoved flag active , since player info buffer,  also has a parameter for position , not only direction
            // since position is essential for MVP matrix (view to be exactly) , we have to recalculate view matrix in MVP and transport it to buffer
            playerMoved = true;
        }

        void movePlayer_right_or_left(bool right , const float dt){
            calcFront();
            glm::vec3 currentPosition = glm::vec3(position);
            const auto rightVector = calcRight();
            // раньше здесь складывался bool `right`, а не вычисленный вектор -> стрейф ехал по диагонали (1,1,1)
            if(right){
                currentPosition += rightVector * dt * speed;
            }else{
                currentPosition -= rightVector * dt * speed;
            }
            position = currentPosition;
            // we will be setting the playerMoved flag active , since player info buffer,  also has a parameter for position , not only direction
            // since position is essential for MVP matrix (view to be exactly) , we have to recalculate view matrix in MVP and transport it to buffer
            playerMoved = true;
        }

        void movePlayer_up_or_left(bool up ,const float dt){
            calcFront();
            if(up){
                position += glm::vec3(0.0f , 1.0f , 0.0) * dt * speed;
            }else{
                position -= glm::vec3(0.0f , 1.0f , 0.0) * dt * speed;
            }
        }

        void assetinput(GLFWwindow* window, const float dt){
            // без этого поворот мышью не применялся, пока не зажат WASD -
            // calcFront() иначе вызывался только изнутри movePlayer_*
            calcFront();
            if(glfwGetKey(window , GLFW_KEY_W) == GLFW_PRESS) movePlayer_straight_or_not(true , dt);
            if(glfwGetKey(window , GLFW_KEY_S) == GLFW_PRESS) movePlayer_straight_or_not(false , dt);
            if(glfwGetKey(window , GLFW_KEY_A) == GLFW_PRESS) movePlayer_right_or_left(false , dt);
            if(glfwGetKey(window , GLFW_KEY_D) == GLFW_PRESS) movePlayer_right_or_left(true , dt);
            if(glfwGetKey(window , GLFW_KEY_SPACE) == GLFW_PRESS) movePlayer_up_or_left(true , dt);
            if(glfwGetKey(window , GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) movePlayer_up_or_left(false , dt);
        }

        // called when the cursor is (re)captured so the first frame after that
        // does not produce a huge delta and snap the camera around
        void resetMouseAnchor() noexcept{
            firstMouseSample = true;
        }

        void processMouseInput(double xpos , double ypos){
            if(firstMouseSample){
                firstMouseSample = false;
                lastX = xpos;
                lastY = ypos;
            }

            const float xoffset = rotation_speed * static_cast<float>(-(lastX - xpos));
            const float yoffset = rotation_speed * static_cast<float>((lastY - ypos));
            yaw += xoffset;
            pitch += yoffset;
            checkPitch();
            lastX = xpos;
            lastY = ypos;
            playerMoved = true;
        }


        static void mousePosCallBack(GLFWwindow* window , double xpos , double ypos){
            if(!window) return;
            Player* user = reinterpret_cast<Player*>(glfwGetWindowUserPointer(window));
            if(user->openedMenu) return;
            user->processMouseInput(xpos , ypos);
        }

        static void keyCallBack(GLFWwindow* window , int key , int scancode , int action, int mode){
            if(!window) return;
            Player* user = reinterpret_cast<Player*>(glfwGetWindowUserPointer(window));
            if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window , true);
            if(key == GLFW_KEY_F && action == GLFW_PRESS){
                user->openedMenu = !user->openedMenu;
                glfwSetInputMode(window , GLFW_CURSOR ,
                                 user->openedMenu ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
                user->resetMouseAnchor();
            }
            if(key == GLFW_KEY_G && action == GLFW_PRESS) user->PT_required = !user->PT_required;
            if(key == GLFW_KEY_R && action == GLFW_PRESS) user->currentMode = (user->currentMode + 1) % user->totalModes;
        }
    };
}