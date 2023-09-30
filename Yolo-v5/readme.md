# Yolo-v5 OpenVINO 异步推理


### 文件结构
- build:编译文件
- data:测试图像图与视频
- include:自定义头文件
- model:模型文件
- opencv:项目所需的opencv头文件与库文件
- openvino:项目所需的openvino头文件与库文件
- resource:md文件资源
- src:源码

### 环境配置
1. OpenCV 4.5.0
   - core:core是opencv的核心组件模块，包括opencv的基础数据类型、数组操作、异步操作、XML/YAML、聚类、一些程序和函数以及宏、OpenGL交互操作、优化算法、eigen支持、opencl支持、并行处理以及硬件加速等等
   - imgproc:imgproc是图像处理方法组件，包括图像滤波、几何图形变换、图像转换、绘图功能、颜色空间转换、直方图、结构分析和形状描述、运动分析和目标跟踪、特征检测、目标检测、图像分割等
   - imgcodecs:用于图像文件的读写
   - videoio:视频的I/O
   - highgui:一些关于创建GUI窗口和鼠标事件的操作等
   - video:运动分析、对象跟踪等
   - calib3d:相机校准和三维重建
   - features2d:二维特征检测描述
   - objdetect:目标检
   - dnn:深度神经网络模
   - ml:机器学习模块
   - flannL:多维空间的聚类和搜索
   - photo:图像修复、去噪等
   - stitching:图像拼接
   - gapi
2. OpenVINO 2023.0.2
3. 异步推理原理
![](resource/async.png)
    进行图片推理的同时完成对上一帧图片的读取
### 网络结构
![](resource/yolov5s.xml.png)
