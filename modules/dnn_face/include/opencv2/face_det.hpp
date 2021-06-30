// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _OPENCV_DNN_FACE_DET_HPP_
#define _OPENCV_DNN_FACE_DET_HPP_

#include <vector>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

/** @defgroup dnn_face DNN used for face recognition
*/

namespace cv
{
namespace dnn_face
{
    //! @addtogroup dnn_face
    //! @{

    /** @brief Structure to hold 5 landmarks for a face bounding box
     */
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

    /** @brief Structure to hold the details pertaining to a face bounding box
     */
    typedef struct Face
    {
        // box of output format: top-left coord (x1, y1), width and height (w, h)
        Rect2i box_tlwh;
        Landmarks_5 landmarks;
        float score;
    } Face;

    /** @brief A class to generate, hold priors and also decode from priors and deltas
     */
    class CV_EXPORTS PriorBox
    {
        public:
            /** @brief Default constructer
            @param shape The size (shape) of input image for generating priors
             */
            PriorBox(const Size& shape);

            /** @brief Decode from priors and deltas (loc, conf, iou)
            @param loc Blob containing relative coordinates of bounding boxes (bboxes) and landmarks
            @param conf Blob containing the probability values of being faces or not
            @param iou Blob containing the IoU values between predicted bboxes and the matched ground truth.
             */
            std::vector<Face> decode(const Mat& loc,
                                     const Mat& conf,
                                     const Mat& iou);

        protected:
            /** @brief Generate priors according to the input shape
             */
            void generate_priors();

        private:
            /** @brief Interger which holds the width of input shape
             */
            int image_width;
            /** @brief Interger which holds the height of input shape
             */
            int image_height;

            /** @brief Vector which holds the feature map sizes of different scales
             */
            std::vector<cv::Size> feature_map_sizes;
            /** @brief Vector which holds the generated priors
             */
            std::vector<Rect2f> priors;
    };

    /** @brief DNN-based face detector and post process wrapper, model download link: https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx.
     */
    class CV_EXPORTS DNNFaceDetector
    {
        public:
            /** @brief Default constructer
            @param onnx_path The path to the face detection model
             */
            DNNFaceDetector(const String& onnx_path);

            /** @brief Face detection
            @param image The input image
            @param score_thresh Threshold for filtering out bboxes with scores smaller than the given value
            @param nms_thresh Threshold for suppressing bboxes with IoU bigger than the given value
            @param top_k Keep top K bboxes before NMS
             */
            std::vector<Face> detect(const Mat& image,
                                     const float score_thresh = 0.6,
                                     const float nms_thresh = 0.3,
                                     const int top_k = 5000);
        protected:
            /** @brief Post-processes including decoding and NMS
            @param shape The shape of input image
            @param loc Blob containing relative coordinates of bounding boxes (bboxes) and landmarks
            @param conf Blob containing the probability values of being faces or not
            @param iou Blob containing the IoU values between predicted bboxes and the matched ground truth.
            @param score_thresh Threshold for filtering out bboxes with scores smaller than the given value
            @param nms_thresh Threshold for suppressing bboxes with IoU bigger than the given value
            @param top_k Keep top K bboxes before NMS
             */
            std::vector<Face> postproc(const Size& shape,
                                       const Mat& loc,
                                       const Mat& conf,
                                       const Mat& iou,
                                       const float score_thresh = 0.6,
                                       const float nms_thresh = 0.3,
                                       const int top_k = 5000);
        private:
            dnn::Net net;
    };

    //! @}
} // namespace dnn_face
} // namespace cv

#endif