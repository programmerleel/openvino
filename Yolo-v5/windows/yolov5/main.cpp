#include "infer.h"

int main() {
	Detector detector;
	detector.model_path = "C:\\Users\\Administrator\\Downloads\\openvino-develop\\Yolo-v5\\model\\yolov5s.xml";
	detector.video_path = "D:\\data\\test_01.mp4";
	detector.new_shape = { 640,640 };
	detector.conf_threshold = 0.25;
	detector.nms_threshold = 0.25;
	infer_sync(detector);

}