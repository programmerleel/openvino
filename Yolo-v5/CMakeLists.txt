cmake_minimum_required(VERSION 3.10)

project(Yolo-v5)

set(OpenCV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv4")
set(OpenVINO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openvino/include")
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${OpenVINO_INCLUDE_DIR})

add_executable(main src/main.cpp)

target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_core.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_dnn.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_highgui.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_imgcodecs.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/libopencv_videoio.so.4.5.0")
target_link_libraries(main "${CMAKE_CURRENT_SOURCE_DIR}/openvino/lib/intel64/libopenvino.so.2023.0.1")