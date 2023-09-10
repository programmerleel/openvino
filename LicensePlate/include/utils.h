#ifndef __HEAD_H__
#define __HEAD_H__

#include <vector>
#include <string>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

void showAvailableDevices(ov::Core core);
cv::Mat cropRoi(cv::Mat image,cv::Rect box,int padding);
cv::Mat resizeImage(std::string image_path,size_t len);
std::vector<cv::Rect> detect(cv::Mat image,ov::Core core);
std::vector<std::string> recognize(std::vector<cv::Rect> boxes,cv::Mat image,ov::Core core);

#endif