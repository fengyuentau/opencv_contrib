#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include <opencv2/face_reg.hpp>
#include <opencv2/face_det.hpp>


using namespace cv;
using namespace std;

Mat getSimilarityTransformMatrix(float src[5][2]) {

    float dst[5][2] = { 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041 };
    float avg0 = (src[0][0] + src[1][0] + src[2][0] + src[3][0] + src[4][0]) / 5;
    float avg1 = (src[0][1] + src[1][1] + src[2][1] + src[3][1] + src[4][1]) / 5;
    //Compute mean of src and dst.
    float src_mean[2] = { avg0, avg1 };
    float dst_mean[2] = { 56.0262, 71.9008 };
    //Subtract mean from src and dst.
    float src_demean[5][2];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            src_demean[j][i] = src[j][i] - src_mean[i];
        }
    }
    float dst_demean[5][2];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            dst_demean[j][i] = dst[j][i] - dst_mean[i];
        }
    }
    double A00 = 0.0, A01 = 0.0, A10 = 0.0, A11 = 0.0;
    for (int i = 0; i < 5; i++)
        A00 += dst_demean[i][0] * src_demean[i][0];
    A00 = A00 / 5;
    for (int i = 0; i < 5; i++)
        A01 += dst_demean[i][0] * src_demean[i][1];
    A01 = A01 / 5;
    for (int i = 0; i < 5; i++)
        A10 += dst_demean[i][1] * src_demean[i][0];
    A10 = A10 / 5;
    for (int i = 0; i < 5; i++)
        A11 += dst_demean[i][1] * src_demean[i][1];
    A11 = A11 / 5;
    Mat A = (Mat_<double>(2, 2) << A00, A01, A10, A11);
    double d[2] = { 1.0, 1.0 };
    double detA = A00 * A11 - A01 * A10;
    if (detA < 0)
        d[1] = -1;
    double T[3][3] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    Mat s, u, vt, v;
    SVD::compute(A, s, u, vt);
    Mat S(s.rows, s.rows, s.type());
    for (int i = 0; i < 2; i++)
        S.ptr<double>(i)[i] = s.ptr<double>(i)[0];
    Mat svd = u * S * vt;
    double smax = s.ptr<double>(0)[0]>s.ptr<double>(1)[0] ? s.ptr<double>(0)[0] : s.ptr<double>(1)[0];
    double tol = smax * 2 * FLT_MIN;
    int rank = 0;
    if (s.ptr<double>(0)[0]>tol)
        rank += 1;
    if (s.ptr<double>(1)[0]>tol)
        rank += 1;
    double arr_u[2][2] = { u.ptr<double>(0)[0], u.ptr<double>(0)[1], u.ptr<double>(1)[0], u.ptr<double>(1)[1] };
    double arr_vt[2][2] = { vt.ptr<double>(0)[0], vt.ptr<double>(0)[1], vt.ptr<double>(1)[0], vt.ptr<double>(1)[1] };
    double det_u = arr_u[0][0] * arr_u[1][1] - arr_u[0][1] * arr_u[1][0];
    double det_vt = arr_vt[0][0] * arr_vt[1][1] - arr_vt[0][1] * arr_vt[1][0];
    if (rank == 1)
    {
        if ((det_u*det_vt) > 0)
        {
            Mat uvt = u*vt;
            T[0][0] = uvt.ptr<double>(0)[0];
            T[0][1] = uvt.ptr<double>(0)[1];
            T[1][0] = uvt.ptr<double>(1)[0];
            T[1][1] = uvt.ptr<double>(1)[1];
        }
        else
        {
            double temp = d[1];
            d[1] = -1;
            Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
            Mat Dvt = D*vt;
            Mat uDvt = u*Dvt;
            T[0][0] = uDvt.ptr<double>(0)[0];
            T[0][1] = uDvt.ptr<double>(0)[1];
            T[1][0] = uDvt.ptr<double>(1)[0];
            T[1][1] = uDvt.ptr<double>(1)[1];
            d[1] = temp;
        }
    }
    else
    {
        Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
        Mat Dvt = D*vt;
        Mat uDvt = u*Dvt;
        T[0][0] = uDvt.ptr<double>(0)[0];
        T[0][1] = uDvt.ptr<double>(0)[1];
        T[1][0] = uDvt.ptr<double>(1)[0];
        T[1][1] = uDvt.ptr<double>(1)[1];
    }
    double var1 = 0.0;
    for (int i = 0; i < 5; i++)
        var1 += src_demean[i][0] * src_demean[i][0];
    var1 = var1 / 5;
    double var2 = 0.0;
    for (int i = 0; i < 5; i++)
        var2 += src_demean[i][1] * src_demean[i][1];
    var2 = var2 / 5;
    double scale = 1.0 / (var1 + var2)* (s.ptr<double>(0)[0] * d[0] + s.ptr<double>(1)[0] * d[1]);
    double TS[2];
    TS[0] = T[0][0] * src_mean[0] + T[0][1] * src_mean[1];
    TS[1] = T[1][0] * src_mean[0] + T[1][1] * src_mean[1];
    T[0][2] = dst_mean[0] - scale*TS[0];
    T[1][2] = dst_mean[1] - scale*TS[1];
    T[0][0] *= scale;
    T[0][1] *= scale;
    T[1][0] *= scale;
    T[1][1] *= scale;
    Mat transform_mat = (Mat_<double>(2, 3) << T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
    return transform_mat;
}

Mat AlignCrop(Mat src_img, dnn_face::Landmarks_5 &face_landmarks) {
    Point2i srcTri[5];
    memcpy(&srcTri, &face_landmarks, 5*sizeof(Point2i));


    float src_point[5][2];
    for (int j = 0; j < 5; j++)
    {
        src_point[j][0] = srcTri[j].x;
        src_point[j][1] = srcTri[j].y;
    }
    Mat warp_mat(2, 3, CV_32FC1);
    warp_mat = getSimilarityTransformMatrix(src_point);
    Mat warp_dst = Mat::zeros(112, 112, src_img.type());
    warpAffine(src_img, warp_dst, warp_mat, warp_dst.size(), INTER_LINEAR);
    return warp_dst;
}


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
    Mat image1 = cv::imread(image1_path);
    Mat image2 = cv::imread(image2_path);

    float score_thresh = 0.9;
    float nms_thresh = 0.3;
    int top_k = 5000;

    // Initialize DNNFaceDetector
    dnn_face::DNNFaceDetector faceDetector(det_onnx_path);
    vector<dnn_face::Face> faces_1 = faceDetector.detect(image1, score_thresh, nms_thresh, top_k);
    vector<dnn_face::Face> faces_2 = faceDetector.detect(image2, score_thresh, nms_thresh, top_k);

    Mat aligned_face1 = AlignCrop(image1, faces_1[0].landmarks);
    Mat aligned_face2 = AlignCrop(image2, faces_2[0].landmarks);

    // // Initialize DNNFaceRecognizer
    dnn_face::DNNFaceRecognizer faceRecognizer(reg_onnx_path);
    Mat feature1 = faceRecognizer.facefeature(aligned_face1);
    Mat feature2 = faceRecognizer.facefeature(aligned_face2);

    float cos_score = faceRecognizer.facematch(feature1, feature2, "cosine");
    float L2_score = faceRecognizer.facematch(feature1, feature2, "norml2");
    
    std::cout<<"Cosine Similarity: "<<cos_score<<endl;
    std::cout<<"NormL2 Distance: "<<L2_score<<endl;

    return 0;
}