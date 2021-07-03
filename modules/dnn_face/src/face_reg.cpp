#include "precomp.hpp"

#include "opencv2/face_reg.hpp"

namespace cv {
namespace dnn_face{

DNNFaceRecognizer::DNNFaceRecognizer(cv::String& onnx_path)
{
    this->model = cv::dnn::readNetFromONNX(onnx_path);
};

cv::Mat DNNFaceRecognizer::facefeature(cv::Mat face_image){
    cv::Mat inputBolb = cv::dnn::blobFromImage(face_image, 1, cv::Size(112, 112), cv::Scalar(0, 0, 0), true, false);
    this->model.setInput(inputBolb);
    return this->model.forward().row(0).clone();
}

float DNNFaceRecognizer::facematch(cv::Mat featureVec1, cv::Mat featureVec2, cv::String distance){
    featureVec1 /= cv::norm(featureVec1);
    featureVec2 /= cv::norm(featureVec2);
    if(distance == "cosine"){
        return cv::sum(featureVec1.mul(featureVec2))[0];
    }else if(distance == "norml2"){
        return cv::norm(featureVec1, featureVec2);
    }else{
        throw std::invalid_argument("invalid parameter " + distance);
    }
}

} // namespace dnn_face
} // namespace cv