#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

std::string data_path = "/home/ubuntu/code/openvino/Yolo-v5/data/city-walk.png";

int main(){
    cv::Mat image = cv::imread(data_path);
    cv::imshow("image",image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    ov::Core core;
}