#ifndef DETECTOR_H
#define DETECTOR_H
 
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#ifdef __cplusplus
extern "C" {
#endif
 
    class Detector{
    private:
        float conf_thred;
        float nms_thred;
    public:
        void set_conf_thred(float conf_thred);
        float get_conf_thred();
        void set_nms_thred(float nms_thred);
        float get_nms_thred();
        std::vector<float> get_paddings(cv::Mat frame,std::vector<float> paddings);
        cv::Mat get_resize_frame(cv::Mat frame,std::vector<float> paddings);
        void async_detect(cv::Mat& paste_frame,ov::InferRequest& request);
    };
    
#ifdef __cplusplus
}
#endif
 
#endif