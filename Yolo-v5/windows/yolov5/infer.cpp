#include "infer.h"

bool infer_sync(Detector detector) {
	// 初始化模型
	detector.init_model(detector.model_path);
	cv::VideoCapture cap(detector.video_path);
	cv::Mat frame;
	std::vector<float> paddings(3);
	DWORD t1, t2;
	while (true)
	{
		t1 = GetTickCount();
		cap.read(frame);
		if (frame.empty())
		{
			break;
		}
		// 前处理
		detector.deal_input(frame, paddings, detector.new_shape);
		auto infer_request = detector.compile_model.create_infer_request();
		// 单线程推理
		auto input_port = detector.compile_model.input();
		ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), detector.blob.ptr(0));
		infer_request.set_input_tensor(input_tensor);
		infer_request.infer();
		auto output = infer_request.get_output_tensor(0);
		cv::Mat output_buffer(output.get_shape()[1], output.get_shape()[2], CV_32F, output.data());
		// 后处理
		detector.deal_output(frame, output_buffer, paddings);
		t2 = GetTickCount();
		std::string fps = std::to_string(1 / ((t2 - t1) * 1.0 / 1000));
		cv::putText(frame, fps, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));
		cv::namedWindow("YOLOv5 OpenVINO Inference C++ Demo", cv::WINDOW_AUTOSIZE);
		cv::imshow("YOLOv5 OpenVINO Inference C++ Demo", frame);
		cv::waitKey(1);
	}
}


