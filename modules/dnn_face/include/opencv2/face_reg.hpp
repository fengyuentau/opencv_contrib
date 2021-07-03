#ifndef __OPENCV_DNN_FACE_REG_HPP__
#define __OPENCV_DNN_FACE_REG_HPP__

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

namespace cv
{ 
namespace dnn_face
{
    class CV_EXPORTS DNNFaceRecognizer
    {
    public:
        /** 
            @brief Default constructer.
            @param onnx_path string& path of the onnx model used for face recognition.
        */
        DNNFaceRecognizer(String& onnx_path);

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
        float facematch(cv::Mat featureVec1, cv::Mat featureVec2, String distance="cosine");
        
    private:
        cv::dnn::Net model;
    };

} // namespace dnn_face
} // namespace cv

#endif // __OPENCV_DNN_FACE_REG_HPP__