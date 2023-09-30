cmake_minimum_required(VERSION 3.10)

project(Yolo-v5)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(OpenVINO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openvino/runtime/cmake")
find_package(OpenVINO REQUIRED)

set(OpenCV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv4")
set(OpenVINO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openvino/runtime/include")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${OpenVINO_INCLUDE_DIR})
include_directories(${INCLUDE_DIR})

add_executable(main src/main.cpp src/detector.cpp)

target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_core.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_dnn.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_highgui.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_imgcodecs.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_videoio.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_imgproc.so.4.5.0")
target_link_libraries(main openvino::runtime)