cmake_minimum_required(VERSION 3.5)

project(LicensePlate)

set(OpenVINO_DIR "/opt/intel/openvino_2023/runtime/cmake")
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

set(OpenVINO_INCLUDE_DIR "/opt/intel/openvino_2023/runtime/include")
set(INCLUDE_DIR "/home/ubuntu/code/openvino/LicensePlate/include")
include_directories(${OpenVINO_INCLUDE_DIR})
include_directories(${INCLUDE_DIR})

add_executable(main src/main.cpp src/utils.cpp)

target_link_libraries(main ${OpenCV_LIBS} openvino::runtime)