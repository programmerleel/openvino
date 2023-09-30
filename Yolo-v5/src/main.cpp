#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <detector.h>

std::string data_path = "/home/ubuntu/code/openvino/Yolo-v5/data/sample.mp4";
std::string model_path = "/home/ubuntu/code/openvino/Yolo-v5/model/yolov5s.xml";
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

int main(){
    //create core
    ov::Core core;
    //read model
    auto read_model = core.read_model(model_path);
    //deal model input
    ov::preprocess::PrePostProcessor pop(read_model);
    ov::preprocess::InputInfo &input_info = pop.input();
    input_info.tensor().set_element_type(ov::element::f32);
    input_info.tensor().set_layout({"NCHW"});
    //compile model
    auto compile_model = core.compile_model(read_model,"CPU");
    //create infer_request 
    ov::InferRequest infer_request = compile_model.create_infer_request();
    ov::InferRequest next_infer_request = compile_model.create_infer_request();
    //create detector
    Detector detector = Detector();
    detector.set_conf_thred(0.25);
    float conf_thred = detector.get_conf_thred();
    detector.set_nms_thred(0.5);
    float nms_thred = detector.get_nms_thred();
    //read frame to get paddings
    cv::Mat frame;
    cv::Mat next_frame;
    cv::VideoCapture cap(data_path);
    std::vector<float> paddings(5);
    cap.read(frame);
    paddings = detector.get_paddings(frame,paddings);
    for (size_t i = 0; i < 5; i++)
    {
        std::cout<<paddings[i]<<std::endl;
    }
    //create callback
    bool ready = false;
    bool next_ready = false;
    std::exception_ptr exception_var;
    //cx,cy,w,h,confidence,c1,c2,...c80
    std::vector<cv::Rect> boxes; //cx,cy,w,h
    std::vector<float> confidences; //conf
    std::vector<float> scores; //c1,c2,...c80 scores
    std::vector<int> ids; //id from the max score in scores
    std::vector<int> indices;
    std::cout<<"pass"<<std::endl;
    infer_request.set_callback([&](std::exception_ptr ex){
        if (ex){
            exception_var = ex;
            return;
        }
        std::cout<<"ha"<<std::endl;
        boxes.clear(); //cx,cy,w,h
        confidences.clear(); //conf
        scores.clear(); //c1,c2,...c80 scores
        ids.clear(); //id from the max score in scores
        indices.clear();
        auto output_tensor = infer_request.get_output_tensor();
        cv::Mat output_data(output_tensor.get_shape()[1],output_tensor.get_shape()[2],CV_32F,(float*)output_tensor.data());
        std::cout<<"ha"<<std::endl;
        //deal model output
        for (size_t i = 0; i < output_tensor.get_shape()[1]; i++){
            float confidence = output_data.at<float>(i,4);
            
            //check box conf
            if (confidence<conf_thred){
                continue;
            }
            cv::Mat scores = output_data.row(i).colRange(5,85);
            cv::Point id;
            //float score; do not use float
            double score;
            cv::minMaxLoc(scores,NULL,&score,NULL,&id);
            //check score
            if (score>0.25){
                //deal box
                float cx = output_data.at<float>(i,0);
                float cy = output_data.at<float>(i,1);
                float w = output_data.at<float>(i,2);
                float h = output_data.at<float>(i,3);
                int left = int((cx-w/2-paddings[1])*paddings[0]);
                int top = int((cy-h/2-paddings[2])*paddings[0]);
                int width = int(w*paddings[0]);
                int height = int(h*paddings[0]);
                cv::Rect box;
                box.x = left;
                box.y = top;
                box.width = width;
                box.height = height;
                boxes.push_back(box);
                confidences.push_back(confidence);
                ids.push_back(id.x);
            }
            //nms
            cv::dnn::NMSBoxes(boxes, confidences, conf_thred, nms_thred, indices);
        }
        ready = true;
        std::cout<<"test"<<std::endl;
    });
    next_infer_request.set_callback([&](std::exception_ptr ex){
        if (ex){
            exception_var = ex;
            return;
        }
        boxes.clear(); //cx,cy,w,h
        confidences.clear(); //conf
        scores.clear(); //c1,c2,...c80 scores
        ids.clear(); //id from the max score in scores
        indices.clear();
        auto output_tensor = next_infer_request.get_output_tensor();
        cv::Mat output_data(output_tensor.get_shape()[2],output_tensor.get_shape()[3],CV_32F,(float*)output_tensor.data());
        //deal model output
        for (size_t i = 0; i < output_tensor.get_shape()[2]; i++){
            float confidence = output_data.at<float>(i,4);
            //check box conf
            if (confidence<conf_thred){
                continue;
            }
            cv::Mat scores = output_data.row(i).colRange(5,85);
            cv::Point id;
            //float score; do not use float
            double score;
            cv::minMaxLoc(scores,NULL,&score,NULL,&id);
            //check score
            if (score>0.25){
                //deal box
                float cx = output_data.at<float>(i,0);
                float cy = output_data.at<float>(i,1);
                float w = output_data.at<float>(i,2);
                float h = output_data.at<float>(i,3);
                int left = int((cx-w/2-paddings[1])*paddings[0]);
                int top = int((cy-h/2-paddings[2])*paddings[0]);
                int width = int(w*paddings[0]);
                int height = int(h*paddings[0]);
                cv::Rect box;
                box.x = left;
                box.y = top;
                box.width = width;
                box.height = height;
                boxes.push_back(box);
                confidences.push_back(confidence);
                ids.push_back(id.x);
            }
            //nms
            cv::dnn::NMSBoxes(boxes, confidences, conf_thred, nms_thred, indices);
        }
        next_ready = true;
    });
    //first detect
    cv::Mat paste_frame = detector.get_resize_frame(frame,paddings);
    std::cout<<paste_frame.size<<std::endl;
    detector.async_detect(paste_frame,infer_request);
    std::cout<<"pass***"<<std::endl;
    while (true){
        bool ret = cap.read(next_frame);
        if (next_frame.empty()){
            break;
        }
        if (ready){
            std::cout<<"oooo"<<std::endl;
            ready = false;
            std::cout<<indices.size()<<std::endl;
            for (size_t i = 0; i < indices.size(); i++){
                int index = indices[i];
                std::cout<<index<<std::endl;
                int id = ids[index];
                std::cout<<id<<std::endl;
                std::cout<<boxes[index]<<std::endl;
                cv::rectangle(frame, boxes[index], colors[id % 6], 2, 8);
                std::cout<<boxes[index]<<std::endl;
                
                std::string label = class_names[id];
                std::cout<<label<<std::endl;
                cv::putText(frame, label, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, colors[id % 6]);
                std::cout<<index<<std::endl;
            }
            cv::imshow("frame",frame);
            next_frame.copyTo(frame);
            paste_frame = detector.get_resize_frame(frame,paddings);
            detector.async_detect(paste_frame,next_infer_request);
        }
        if (next_ready){
            next_ready = false;
            for (size_t i = 0; i < indices.size(); i++){
                int index = indices[i];
                int id = ids[index];
                cv::rectangle(frame, boxes[index], colors[id % 6], 2, 8);
                std::string label = class_names[id];
                cv::putText(frame, label, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, colors[id % 6]);
            }
            cv::imshow("frame",frame);
            next_frame.copyTo(frame);
            paste_frame = detector.get_resize_frame(frame,paddings);
            detector.async_detect(paste_frame,infer_request);
        }
        char c = cv::waitKey(1);
		if (c == 27) { // ESC
			break;
		}
    } 
    cv::waitKey(0);
    cv::destroyAllWindows();
}