cmake_minimum_required(VERSION 3.5)
project(deep_sea)
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(bitbots_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(control_toolbox REQUIRED)
find_package(backward_ros REQUIRED)

# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable
add_executable(sea_correction_helper_cpp src/sea_correction_helper.cpp)
ament_target_dependencies(sea_correction_helper_cpp rclcpp
        bitbots_msgs
        pluginlib
        rosidl_typesupport_cpp
        std_msgs
        std_srvs
        control_msgs
        control_toolbox
        backward_ros)

# Install Cpp executables
install(TARGETS
        sea_correction_helper_cpp
        DESTINATION lib/${PROJECT_NAME}
        )

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
        deep_sea/sea_correction_helper.py
        DESTINATION lib/${PROJECT_NAME}
        )

install(DIRECTORY scripts/
        DESTINATION lib/${PROJECT_NAME})

ament_package()