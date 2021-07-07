# Face Regconition using Convolutional Neural Networks

This module uses Convolutional Neural Networks for face detection and regconition.

## Dependencies
- OpenCV DNN Module

## Build

Run the following command to build this module with OpenCV:
```shell
cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_dir>
```

## Models

There are two models (ONNX format) pre-trained and required for this module:
- [Face Detection](https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx):
    - Size: 337KB
    - Results on WIDER Face Val set: 0.830(easy), 0.824(medium), 0.708(hard)
- [Face Recognition](https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view?usp=sharing)

## Examples

### Face Detection
```cpp
// Initialize DNNFaceDetector with onnx_path (cv::String)
cv::dnn_face::DNNFaceDetector faceDetector(onnx_path);

// Run face detection with given image (cv::Mat), score_thresh (float), nms_thresh (float) and top_k (int)
std::vector<dnn_face::Face> faces = faceDetector.detect(image, score_thresh, nms_thresh, top_k);
```

`dnn_face::Face` is a struct holding a `cv::Rect2i` for face bounding boxes (top-left coordinates, width, height), a struct (`dnn_face::Landmarks_5`) variable for landmarks and a detection score (`float`).
```cpp
typedef struct Face
{
    cv::Rect2i box_tlwh;
    dnn_face::Landmarks_5 landmarks;
    float score;
} Face;
```

`dnn_face::Landmarks_5` holds the coordinates of 5 landmarks, which are right and left eyes, nose tip, the right and left corners of the mouth.
```cpp
typedef struct Landmarks_5
{
    cv::Point2i right_eye;
    cv::Point2i left_eye;
    cv::Point2i nose_tip;
    cv::Point2i mouth_right;
    cv::Point2i mouth_left;
} Landmarks_5;
```

### Face Recognition
