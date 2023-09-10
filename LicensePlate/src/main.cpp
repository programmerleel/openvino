#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "utils.h"

std::string image_path = "/home/ubuntu/code/openvino/LicensePlate/image/car_test.bmp";

int main(){
    //create core
    ov::Core core;
    //show available devices
    showAvailableDevices(core);
    //read image
    cv::Mat image = cv::imread(image_path);
    //detect
    std::vector<cv::Rect> boxes = detect(image,core);
    //recognize
    std::vector<std::string> results = recognize(boxes,image,core);
    for (size_t i = 0; i < boxes.size(); i++){
        cv::Rect box = boxes[i];
        cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8, 0);
        std::string result = results[i];
        cv::putText(image, result.c_str(), cv::Point(box.x - 50, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2, 8);
    }
	cv::imshow("image", image);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
