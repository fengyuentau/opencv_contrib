// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_FACE_REG_HPP__
#define __OPENCV_DNN_FACE_REG_HPP__

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "face_def.hpp"

namespace cv
{ 
namespace dnn_face
{
    class CV_EXPORTS DNNFaceRecognizer
    {
    public:
        /** 
            @brief Default constructer.
            @param onnx_path const String& The path of the onnx model used for face recognition.
        */
        DNNFaceRecognizer(const String& onnx_path);

        /** 
            @brief Extracting face feature.
            @param face_image Mat The input face.
        */
        Mat facefeature(Mat face_image);

        /** 
            @brief Calculating the distance between two face features.
            @param featureVec1 Mat The first input feature.
            @param featureVec2 Mat The second input feature of the same size and the same type as src1.
            @param distance const String& Defining the similarity with optional values "norml2" or "cosine".
        */
        float facematch(Mat featureVec1, Mat featureVec2, const String& distance="cosine");
        Mat AlignCrop(Mat src_img, Landmarks_5 &face_landmarks);
    private:
        dnn::Net model;
        Mat getSimilarityTransformMatrix(float src[5][2]);
    };

} // namespace dnn_face
} // namespace cv

#endif // __OPENCV_DNN_FACE_REG_HPP__