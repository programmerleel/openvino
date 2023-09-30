#include "detector.h"
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

void Detector::set_conf_thred(float conf){
    Detector::conf_thred = conf;
}

float Detector::get_conf_thred(){
    return Detector::conf_thred;
}

void Detector::set_nms_thred(float conf){
    Detector::nms_thred = conf;
}

float Detector::get_nms_thred(){
    return Detector::nms_thred;
}

std::vector<float> Detector::get_paddings(cv::Mat frame,std::vector<float> paddings){
    int cols = frame.cols;
    int rows = frame.rows;
    int max_len = MAX(cols,rows);
    float ratio = max_len/640.0;
    paddings[0] = ratio;
    int w = int(cols/ratio);
    int h = int(rows/ratio);
    float padding_w = (640.0-w)/2.0;
    float padding_h = (640.0-h)/2.0;
    paddings[1] = padding_w;
    paddings[2] = padding_h;
    paddings[3] = w;
    paddings[4] = h;
    return paddings;
}

cv::Mat Detector::get_resize_frame(cv::Mat frame,std::vector<float> paddings){
    int w = paddings[3];
    int h = paddings[4];
    int top = int(paddings[2]);
    int bottom = int(paddings[2]);
    int left = int(paddings[1]);
    int right = int(paddings[1]);
    cv::Mat resize_frame;
    cv::Mat paste_frame;
    cv::resize(frame,resize_frame,cv::Size(w,h));
    cv::copyMakeBorder(resize_frame,paste_frame,top,bottom,left,right,0,cv::Scalar(0,0,0));
    return paste_frame;
}

void Detector::async_detect(cv::Mat& paste_frame,ov::InferRequest& request){
    paste_frame.convertTo(paste_frame,CV_32F);
    paste_frame = paste_frame/255.0;
    std::cout<<paste_frame.size<<std::endl;
    ov::Tensor input_tensor = request.get_input_tensor();
    int w = input_tensor.get_shape()[3];
    int h = input_tensor.get_shape()[2];
    int c = input_tensor.get_shape()[1];
    std::cout<<w<<std::endl;
    std::cout<<h<<std::endl;
    std::cout<<c<<std::endl;
    int frame_size = w*h;
    float* data = input_tensor.data<float>();
    for (size_t i = 0; i < h; i++){
        for (size_t j = 0; j < w; j++){
            for (size_t k = 0; k < c; k++){
                data[frame_size*k+i*w+j] = paste_frame.at<cv::Vec3f>(i,j)[k];
            }
        } 
    }
    std::cout<<"ssap"<<std::endl;
    request.start_async();
    std::cout<<"passpass"<<std::endl;
}