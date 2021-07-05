// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_DNN_FACE_DEF_HPP_
#define _OPENCV_DNN_FACE_DEF_HPP_

#include <opencv2/imgproc.hpp>

/** @defgroup dnn_face Defninitions of faces and landmarks
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

    //! @}
} // namespace dnn_face
} // namespace cv

#endif
