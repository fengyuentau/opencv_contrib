#ifndef __OPENCV_DNN_FACE_REG_HPP__
#define __OPENCV_DNN_FACE_REG_HPP__

#include "precomp.hpp"

namespace cv { 
namespace dnn_face_reg{

class CV_EXPORTS_W DNNFaceRecognizer
{
public:
    DNNFaceRecognizer(string& onnx_path);
    cv::Mat facefeature(cv::Mat face_image);
    float facematch(cv::Mat featureVec1, cv::Mat featureVec2);
private:
    cv::dnn::Net model;
};

} // namespace dnn_face_reg
} // namespace cv