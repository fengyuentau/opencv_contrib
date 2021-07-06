// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include <opencv2/face_reg.hpp>
#include <opencv2/face_det.hpp>


using namespace cv;
using namespace std;


int main(int argc, char ** argv)
{
    if (argc != 5)
    {
        std::cerr << "Usage " << argv[0] << ": "
                  << "<det_onnx_path> "
                  << "<reg_onnx_path> "
                  << "<image1>"
                  << "<image2>\n";
        return -1;
    }

    String det_onnx_path = argv[1];
    String reg_onnx_path = argv[2];
    String image1_path = argv[3];
    String image2_path = argv[4];
    std::cout<<image1_path<<" "<<image2_path<<std::endl;
    Mat image1 = imread(image1_path);
    Mat image2 = imread(image2_path);

    float score_thresh = 0.9;
    float nms_thresh = 0.3;
    float cosine_similar_thresh = 0.34;
    float l2norm_similar_thresh = 1.32;
    int top_k = 5000;

    // Initialize DNNFaceDetector
    dnn_face::DNNFaceDetector faceDetector(det_onnx_path);
    vector<dnn_face::Face> faces_1 = faceDetector.detect(image1, score_thresh, nms_thresh, top_k);
    if (faces_1.size() < 1)
    {
        std::cerr << "Cannot find a face in " << image1_path << "\n";
        return -1;
    }
    vector<dnn_face::Face> faces_2 = faceDetector.detect(image2, score_thresh, nms_thresh, top_k);
    if (faces_2.size() < 1)
    {
        std::cerr << "Cannot find a face in " << image2_path << "\n";
        return -1;
    }

    // Initialize DNNFaceRecognizer
    dnn_face::DNNFaceRecognizer faceRecognizer(reg_onnx_path);

    Mat aligned_face1 = faceRecognizer.AlignCrop(image1, faces_1[0].landmarks);
    Mat aligned_face2 = faceRecognizer.AlignCrop(image2, faces_2[0].landmarks);

    Mat feature1 = faceRecognizer.facefeature(aligned_face1);
    Mat feature2 = faceRecognizer.facefeature(aligned_face2);

    float cos_score = faceRecognizer.facematch(feature1, feature2, "cosine");
    float L2_score = faceRecognizer.facematch(feature1, feature2, "norml2");
    
    if(cos_score >= cosine_similar_thresh){
        std::cout<<"They have the same identity."<<endl;
    }else{
        std::cout<<"They have different identities."<<endl;
    }

    if(L2_score <= l2norm_similar_thresh){
        std::cout<<"They have the same identity."<<endl;
    }else{
        std::cout<<"They have different identities."<<endl;
    }
    
    std::cout<<"Cosine Similarity: "<<cos_score<<endl;
    std::cout<<"NormL2 Distance: "<<L2_score<<endl;

    return 0;
}