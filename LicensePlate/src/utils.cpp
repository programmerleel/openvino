#include "utils.h"

//show available devices
void showAvailableDevices(ov::Core core){
    std::vector<std::string> availableDevices = core.get_available_devices();
    for (size_t i = 0; i < availableDevices.size(); i++){
        std::cout<<"surported device name:"<<availableDevices[i]<<std::endl;
    }
}

//crop roi
cv::Mat cropRoi(cv::Mat image,cv::Rect box,int padding){
    cv::Rect roi;
    roi.x = box.x-padding;
    roi.y = box.y-padding;
    roi.width = box.width+2*padding;
    roi.height = box.height+padding*2;
    cv::Mat roi_image = image(roi);
    return roi_image;
}

//euqal scale image
cv::Mat resizeImage(std::string image_path,size_t len){
    cv::Mat image = cv::imread(image_path);
    size_t rows = image.rows;
    size_t cols = image.cols;
    size_t max_len = MAX(rows,cols);
    float ratio = float(len)/float(max_len);
    int scale_rows = (int)rows*ratio;
    int scale_cols = (int)cols*ratio;
    cv::resize(image,image,cv::Size(scale_cols,scale_rows));
    int top = (int)(rows-scale_rows)/2;
    int bottom = (int)(rows-scale_rows)/2;
    int left = (int)(cols-scale_cols)/2;
    int right = (int)(cols-scale_cols)/2;
    cv::Mat blob;
    cv::copyMakeBorder(image, blob, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return blob;
}

std::string detect_model_path = "/home/ubuntu/code/openvino/LicensePlate/model/vehicle-license-plate-detection-barrier-0106.xml";

std::vector<cv::Rect> detect(cv::Mat image,ov::Core core){
    //read model
    auto model = core.read_model(detect_model_path);
    //deal model input
    ov::preprocess::PrePostProcessor pop(model);
    ov::preprocess::InputInfo& inputInfo = pop.input();
    inputInfo.tensor().set_element_type(ov::element::u8);
    inputInfo.tensor().set_layout( {"NCHW"} );
    //make deal
    model = pop.build();
    ov::CompiledModel compile_model = core.compile_model(model,"CPU");
    //compile model
    ov::InferRequest inferRequest = compile_model.create_infer_request();
    //deal input image
    ov::Shape input_shape = inferRequest.get_input_tensor().get_shape();
    size_t c = input_shape[1];
    size_t h = input_shape[2];
    size_t w = input_shape[3];
    size_t rows = image.rows;
    size_t cols = image.cols;
    cv::Mat blob;
    cv::resize(image,blob,cv::Size(w,h));
    cv::Mat input_blob = cv::dnn::blobFromImage(blob,(1.0),cv::Size(),cv::Scalar(),false,false,CV_8U);
    auto input_port = compile_model.input();
    //set input
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), input_blob.ptr(0));
    inferRequest.set_input_tensor(input_tensor);
    //infer
    inferRequest.infer();
    //deal output
    ov::Tensor output_tensor = inferRequest.get_output_tensor();
	cv::Mat prob(output_tensor.get_shape()[2], output_tensor.get_shape()[3], CV_32F, (float*)output_tensor.data());
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < output_tensor.get_shape()[2]; i++) {
        int label_id = prob.at<float>(i,1);
        float conf = prob.at<float>(i,2);
        if (label_id == 2){
            if (conf > 0.75) {
			int x_min = static_cast<int>(prob.at<float>(i, 3)*cols);
			int y_min = static_cast<int>(prob.at<float>(i, 4)*rows);
			int x_max = static_cast<int>(prob.at<float>(i, 5)*cols);
			int y_max = static_cast<int>(prob.at<float>(i, 6)*rows);
			cv::Rect box(x_min, y_min, x_max - x_min, y_max - y_min);
            boxes.push_back(box);
            }
        }
	}
    return boxes;
}

static const char* const items[] = {
				"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
				"<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
				"<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
				"<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
				"<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
				"<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
				"<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
				"<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
				"<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
				"<Zhejiang>", "<police>",
				"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
				"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
				"U", "V", "W", "X", "Y", "Z"
};
std::string recognize_model_path = "/home/ubuntu/code/openvino/LicensePlate/model/license-plate-recognition-barrier-0001.xml";

std::vector<std::string> recognize(std::vector<cv::Rect> boxes,cv::Mat image,ov::Core core){
    //read model
    auto model = core.read_model(recognize_model_path);
    //deal model input
    ov::OutputVector inputs = model->inputs();
    std::string m_LprInputName;
    std::string m_LprInputSeqName;
    size_t m_LprInputId;
    for (auto input:inputs){
        if (input.get_shape().size() == 4){ 
			m_LprInputName = input.get_any_name();
            m_LprInputId = input.get_index();
            }
		if (input.get_shape().size() == 2){
			m_LprInputSeqName = input.get_any_name();
            }
    }
    ov::preprocess::PrePostProcessor pop(model);
    ov::preprocess::InputInfo& inputInfo = pop.input(m_LprInputName);
    inputInfo.tensor().set_element_type(ov::element::u8);
    inputInfo.tensor().set_layout( {"NCHW"} );
    //make deal
    model = pop.build();
    ov::CompiledModel compile_model = core.compile_model(model,"CPU");
    //compile model
    ov::InferRequest inferRequest = compile_model.create_infer_request();
    //deal input image
    ov::Shape input_shape = inferRequest.get_tensor(m_LprInputName).get_shape();
    ov::Shape input_seq_shape = inferRequest.get_tensor(m_LprInputSeqName).get_shape();
    size_t c = input_shape[1];
    size_t h = input_shape[2];
    size_t w = input_shape[3];
    std::vector<std::string> results;
    for (size_t i = 0; i < boxes.size(); i++){
        cv::Rect box = boxes[i];
        cv::Mat roi_image = cropRoi(image,box,5);
        cv::Mat blob;
        cv::resize(roi_image,blob,cv::Size(w,h));
        cv::Mat input_blob = cv::dnn::blobFromImage(blob,(1.0),cv::Size(),cv::Scalar(),false,false,CV_8U);
        cv::Mat input_seq = cv::Mat::ones(input_seq_shape[0], input_seq_shape[1], CV_32F);
        //set input
        auto input_port = compile_model.input(m_LprInputId);
        auto input_seq_port = compile_model.input(1);
        ov::Tensor input_tensor(input_port.get_element_type(),input_port.get_shape(),input_blob.ptr(0));
        ov::Tensor input_seq_tensor(input_seq_port.get_element_type(),input_seq_port.get_shape(),input_seq.ptr(0));
        inferRequest.set_input_tensor(m_LprInputId,input_tensor);
        inferRequest.set_input_tensor(1,input_seq_tensor);
        //infer
        inferRequest.infer();
        //deal output
        ov::Tensor output_tensor = inferRequest.get_output_tensor();
	    cv::Mat prob(output_tensor.get_shape()[1], output_tensor.get_shape()[2], CV_32F, (float*)output_tensor.data());
        std::string result;
        result.reserve(14u + 6u);
        for (int i = 0; i < output_tensor.get_shape()[1]; i++){
		    int val = static_cast<int>(prob.at<float>(i, 0));
		    if (val == -1) {
			    break;
		    }
		    result += items[val];
	    }
        results.push_back(result);
    }
    return results;
}