#ifndef __OPENCV_DNN_FACE_REG_HPP__
#define __OPENCV_DNN_FACE_REG_HPP__

#include "precomp.hpp"

namespace cv { 
namespace dnn_face_reg{

class CV_EXPORTS_W DNNFaceRecognizer
{
public:
    /** 
        @brief Default constructer.
        @param onnx_path string& path of the onnx model used for face recognition.
    */
    DNNFaceRecognizer(string& onnx_path);

    /** 
        @brief Extracting face feature.
        @param face_image cv::Mat input face.
    */
    cv::Mat facefeature(cv::Mat face_image);

    /** 
        @brief Calculating the distance between two face feature.
        @param featureVec1 cv::Mat first input feature.
        @param featureVec2 cv::Mat second input feature of the same size and the same type as src1.
        @param distance string defining the similarity with optional values "norml2" and "cosine"
    */
    float facematch(cv::Mat featureVec1, cv::Mat featureVec2, string distance="cosine");
    
private:
    cv::dnn::Net model;
};

} // namespace dnn_face_reg
} // namespace cv