#ifndef DETECTOR_H

#define DETECTOR_H

#include <opencv.hpp>
#include <openvino.hpp>
#include <iostream>
#include <Windows.h>

using std::min;

class Detector
{
// 搬到共有 正常应该写 set get 函数
//private:
//	std::string model_path;
//	std::string video_path;
//
//	std::shared_ptr<ov::Model> read_model;
//	ov::CompiledModel compile_model;
//
//	std::vector<int> new_shape;
//	cv::Mat blob;
//
//	std::vector<int> indices;
//
//	double conf_threshold;
//	double nms_threshold;

public:
	std::string model_path;
	std::string video_path;

	ov::CompiledModel compile_model;

	std::vector<int> new_shape;
	cv::Mat blob;

	double conf_threshold;
	double nms_threshold;
	typedef struct {
		cv::Rect rect;
		cv::Point class_id;
		double class_score;
	} Object;

	// 初始化mdoel
	bool init_model(std::string xml_path);
	// 前处理
	bool deal_input(cv::Mat& img, std::vector<float>& paddings, std::vector<int> new_shape);
	// 后处理
	bool deal_output(cv::Mat& frame, cv::Mat& output_buffer, std::vector<float>& paddings);
	// 画图
	bool draw(cv::Mat& frame, std::vector<std::string>& class_names, std::vector<cv::Scalar> colors, int class_id, float class_score, cv::Rect box);
};

#endif
