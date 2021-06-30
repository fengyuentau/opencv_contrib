// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _OPENCV_DNN_FACE_DET_HPP_
#define _OPENCV_DNN_FACE_DET_HPP_

#include <vector>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn_face
{
    typedef struct Landmarks_5
    {
        // right eye
        Point2i right_eye;
        // left eye
        Point2i left_eye;
        // nose
        Point2i nose_tip;
        // mouth right
        Point2i mouth_right;
        // mouth left
        Point2i mouth_left;
    } Landmarks_5;

    typedef struct Face
    {
        // box of output format: top-left coord (x1, y1), width and height (w, h)
        Rect2i box_tlwh;
        Landmarks_5 landmarks;
        float score;
    } Face;

    class CV_EXPORTS PriorBox
    {
        public:
            PriorBox(const Size& shape);
            std::vector<Face> decode(const Mat& loc,
                                     const Mat& conf,
                                     const Mat& iou);

        protected:
            void generate_priors();

        private:
            int image_width;
            int image_height;

            std::vector<cv::Size> feature_map_sizes;
            std::vector<Rect2f> priors;
    };

    class CV_EXPORTS DNNFaceDetector
    {
        // download the face detection model (ONNX) at the following link:
        // https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx
        public:
            DNNFaceDetector(const String& onnx_path);
            std::vector<Face> detect(const Mat& image,
                                     const float score_thresh = 0.6,
                                     const float nms_thresh = 0.3,
                                     const int top_k = 5000);
        protected:
            std::vector<Face> postproc(const Size& shape,
                                       const Mat& loc,
                                       const Mat& iou,
                                       const Mat& conf,
                                       const float score_thresh = 0.6,
                                       const float nms_thresh = 0.3,
                                       const int top_k = 5000);
        private:
            dnn::Net net;
    };
}
}

#endif