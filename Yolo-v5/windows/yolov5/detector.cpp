#include "detector.h"

std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
								   cv::Scalar(255, 255, 0) , cv::Scalar(0, 255, 255) , cv::Scalar(255, 0, 255) };
std::vector<std::string> class_names = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush" };

bool Detector::init_model(std::string xml_path) {
	ov::Core core;
	auto read_model = core.read_model(Detector::model_path);
	ov::preprocess::PrePostProcessor pop(read_model);
	ov::preprocess::InputInfo& input_info = pop.input();
	// 对模型进行预处理 加快推理速度
	input_info.tensor().set_layout({ "NCHW" });
	// 模型输入改为u8
	// 这里一点疑惑 将模型输入改为u8 不对输入进行归一化 但是产生的结果非常差
	//input_info.tensor().set_element_type(ov::element::u8);
	input_info.tensor().set_element_type(ov::element::f32);
	Detector::compile_model = core.compile_model(read_model, "CPU");
}


bool Detector::deal_input(cv::Mat& img, std::vector<float>& paddings, std::vector<int> new_shape) {
	int img_h = img.rows;
	int img_w = img.cols;
	float scale = min(new_shape[1] * 1.0 / img_h, new_shape[0] * 1.0 / img_w);
	int resize_h = int(round(img_h * scale));
	int resize_w = int(round(img_w * scale));
	paddings[0] = scale;

	int pad_h = new_shape[1] - resize_h;
	int pad_w = new_shape[0] - resize_w;

	cv::Mat letter_img;
	cv::resize(img, letter_img, cv::Size(resize_w, resize_h));

	float half_h = pad_h * 1.0 / 2;
	float half_w = pad_w * 1.0 / 2;
	paddings[1] = half_h;
	paddings[2] = half_w;

	int top = int(round(half_h - 0.1));
	int bottom = int(round(half_h + 0.1));
	int left = int(round(half_w - 0.1));
	int right = int(round(half_w + 0.1));

	cv::copyMakeBorder(letter_img, letter_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
	
	Detector::blob = cv::dnn::blobFromImage(letter_img, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0));
}

bool Detector::deal_output(cv::Mat& frame,cv::Mat& output_buffer, std::vector<float>& paddings) {
	std::vector<Detector::Object> results;
	std::vector<cv::Rect> boxes;
	std::vector<int> class_ids;
	std::vector<float> class_scores;
	std::vector<float> confidences;
	// cx,cy,w,h,confidence,c1,c2,...c80
	for (int i = 0; i < output_buffer.rows; i++) {
		Detector::Object result;
		float confidence = output_buffer.at<float>(i, 4);
		// 通过置信度排除一部分 减轻后续处理压力
		if (confidence < conf_threshold) {
			continue;
		}
		cv::Mat classes_scores = output_buffer.row(i).colRange(5, 85);
		cv::minMaxLoc(classes_scores, NULL, &result.class_score, NULL, &result.class_id);

		if (result.class_score > 0.25){
			// box坐标
			float cx = output_buffer.at<float>(i, 0);
			float cy = output_buffer.at<float>(i, 1);
			float w = output_buffer.at<float>(i, 2);
			float h = output_buffer.at<float>(i, 3);
			// 恢复原图比例
			int left = static_cast<int>((cx - 0.5 * w - paddings[2]) / paddings[0]);
			int top = static_cast<int>((cy - 0.5 * h - paddings[1]) / paddings[0]);
			int width = static_cast<int>(w / paddings[0]);
			int height = static_cast<int>(h / paddings[0]);
			cv::Rect box;
			box.x = left;
			box.y = top;
			box.width = width;
			box.height = height;
			result.rect = box;
			boxes.push_back(box);
			class_ids.push_back(result.class_id.x);
			class_scores.push_back(result.class_score);
			confidences.push_back(confidence);
			results.push_back(result);
		}
	}

	// NMS
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, Detector::conf_threshold, Detector::nms_threshold, indices);
	for (size_t i = 0; i < indices.size(); i++) {
		int index = indices[i];
		Detector::Object result = results[index];
		int id = result.class_id.x;
		float score = result.class_score;
		cv::Rect box = result.rect;
		Detector::draw(frame, class_names, colors, id, score, box);
	}
}

bool Detector::draw(cv::Mat& frame,std::vector<std::string>& class_names, std::vector<cv::Scalar> colors,int class_id,float class_score,cv::Rect box) {
	cv::rectangle(frame, box, colors[class_id % 6], 2, 8);
	std::string label = class_names[class_id] + ":" + std::to_string(class_score);
	cv::putText(frame, label, cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, colors[class_id % 6]);
}