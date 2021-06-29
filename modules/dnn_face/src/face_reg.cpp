#ifndef __OPENCV_DNN_FACE_REG_HPP__
#define __OPENCV_DNN_FACE_REG_HPP__

#include "precomp.hpp"
#include <opencv2/face_reg.hpp>

namespace cv {
namespace dnn_face_reg{

DNNFaceRecognizer::DNNFaceRecognizer(string& onnx_path)
{
    this->model = cv::dnn::readNetFromONNX(onnx_path);
};

cv::Mat DNNFaceRecognizer::facefeature(cv::Mat face_image){
    cv::Mat inputBolb = cv::dnn::blobFromImage(face_image, 1, cv::Size(112, 112), cv::Scalar(0, 0, 0), true, false);
    this->model.setInput(inputBolb);
    return this->model.forward()[0];
}

float DNNFaceRecognizer::facematch(cv::Mat featureVec1, cv::Mat featureVec2, string distance);{
    if(distance == "cosine"){
        return cv::sum(featureVec1.mul(featureVec2))[0]/(cv::norm(featureVec1)*cv::norm(featureVec2));
    }else if(distance == "norml2"){
        return cv::norm(featureVec1/cv::norm(featureVec1), featureVec2/cv::norm(featureVec2));
    }else{
        throw std::invalid_argument("invalid parameter " + distance);
    }
}

} // namespace dnn_face_reg
} // namespace cv