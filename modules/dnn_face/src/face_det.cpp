// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/face_det.hpp"

namespace cv
{
namespace dnn_face
{
    PriorBox::PriorBox(const Size& shape)
    {
        image_width = shape.width;
        image_height = shape.height;

        // Calculate shapes of different scales according to the shape of input image
        Size feature_map_2nd = {
            int(int((image_width+1)/2)/2), int(int((image_height+1)/2)/2)
        };
        Size feature_map_3rd = {
            int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
        };
        Size feature_map_4th = {
            int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
        };
        Size feature_map_5th = {
            int(feature_map_4th.width/2), int(feature_map_4th.height/2)
        };
        Size feature_map_6th = {
            int(feature_map_5th.width/2), int(feature_map_5th.height/2)
        };

        feature_map_sizes.push_back(feature_map_3rd);
        feature_map_sizes.push_back(feature_map_4th);
        feature_map_sizes.push_back(feature_map_5th);
        feature_map_sizes.push_back(feature_map_6th);

        // Generate priors
        generate_priors();
    }

    void PriorBox::generate_priors()
    {
        // Fixed params for generating priors
        const std::vector<std::vector<float>> min_sizes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}
        };
        const std::vector<int> steps = { 8, 16, 32, 64 };

        // Generate priors
        for (size_t i = 0; i < feature_map_sizes.size(); ++i)
        {
            Size feature_map_size = feature_map_sizes[i];
            std::vector<float> min_size = min_sizes[i];

            for (int _h = 0; _h < feature_map_size.height; ++_h)
            {
                for (int _w = 0; _w < feature_map_size.width; ++_w)
                {
                    for (size_t j = 0; j < min_size.size(); ++j)
                    {
                        float s_kx = min_size[j] / image_width;
                        float s_ky = min_size[j] / image_height;

                        float cx = (_w + 0.5) * steps[i] / image_width;
                        float cy = (_h + 0.5) * steps[i] / image_height;

                        Rect2f prior = { cx, cy, s_kx, s_ky };
                        priors.push_back(prior);
                    }
                }
            }
        }
    }

    std::vector<Face> PriorBox::decode(const Mat& loc,
                             const Mat& conf,
                             const Mat& iou)
    {
        const std::vector<float> variance = {0.1, 0.2};

        // num * [bbox (Rect2i), 5-landmarks (Landmarks_5), score (float)]
        std::vector<Face> dets;

        float* loc_v = (float*)(loc.data);
        float* conf_v = (float*)(conf.data);
        float* iou_v = (float*)(iou.data);
        for (size_t i = 0; i < priors.size(); ++i) {
            Face face;

            // Get score
            float cls_score = conf_v[i*2+1];
            float iou_score = iou_v[i];
            // Clamp
            if (iou_score < 0.f) {
                iou_score = 0.f;
            }
            else if (iou_score > 1.f) {
                iou_score = 1.f;
            }
            float score = std::sqrt(cls_score * iou_score);
            face.score = score;

            // Get bounding box
            float cx = (priors[i].x + loc_v[i*14+0] * variance[0] * priors[i].width) * image_width;
            float cy = (priors[i].y + loc_v[i*14+1] * variance[0] * priors[i].height) * image_height;
            float w  = priors[i].width * exp(loc_v[i*14+2] * variance[0]) * image_width;
            float h  = priors[i].height * exp(loc_v[i*14+3] * variance[1]) * image_height;
            int x1 = int(cx - w / 2);
            int y1 = int(cy - h / 2);
            face.box_tlwh = { x1, y1, int(w), int(h) };

            // Get landmarks
            int x_re = int((priors[i].x + loc_v[i*14+ 4] * variance[0] * priors[i].width)  * image_width);
            int y_re = int((priors[i].y + loc_v[i*14+ 5] * variance[0] * priors[i].height) * image_height);
            int x_le = int((priors[i].x + loc_v[i*14+ 6] * variance[0] * priors[i].width)  * image_width);
            int y_le = int((priors[i].y + loc_v[i*14+ 7] * variance[0] * priors[i].height) * image_height);
            int x_n = int((priors[i].x + loc_v[i*14+ 8] * variance[0] * priors[i].width)  * image_width);
            int y_n = int((priors[i].y + loc_v[i*14+ 9] * variance[0] * priors[i].height) * image_height);
            int x_mr  = int((priors[i].x + loc_v[i*14+10] * variance[0] * priors[i].width)  * image_width);
            int y_mr  = int((priors[i].y + loc_v[i*14+11] * variance[0] * priors[i].height) * image_height);
            int x_ml = int((priors[i].x + loc_v[i*14+12] * variance[0] * priors[i].width)  * image_width);
            int y_ml = int((priors[i].y + loc_v[i*14+13] * variance[0] * priors[i].height) * image_height);
            face.landmarks = {
                {x_re, y_re}, // right eye
                {x_le, y_le}, // left eye
                {x_n,  y_n }, // nose
                {x_mr, y_mr}, // mouth right
                {x_ml, y_ml}  // mouth left
            };

            dets.push_back(face);
        }

        return dets;
    }

    DNNFaceDetector::DNNFaceDetector(const String& onnx_path)
    {
        net = dnn::readNet(onnx_path);
    }

    std::vector<Face> DNNFaceDetector::detect(const Mat& image,
                                              const float score_thresh,
                                              const float nms_thresh,
                                              const int top_k)
    {
        // Build blob from image
        Mat blob = dnn::blobFromImage(image);

        // Forward
        std::vector<String> output_names = { "loc", "conf", "iou" };
        std::vector<Mat> output_blobs;
        net.setInput(blob);
        net.forward(output_blobs, output_names);

        // Post process
        std::vector<Face> faces = postproc(image.size(), output_blobs[0], output_blobs[1], output_blobs[2], score_thresh, nms_thresh, top_k);

        return faces;
    }

    std::vector<Face> DNNFaceDetector::postproc(const Size& shape,
                                                const Mat& loc,
                                                const Mat& conf,
                                                const Mat& iou,
                                                const float score_thresh,
                                                const float nms_thresh,
                                                const int top_k)
    {
        // Decode from priorbox and deltas
        PriorBox pb(shape);
        std::vector<Face> faces = pb.decode(loc, conf, iou);

        // Perform NMS
        if (faces.size() > 1)
        {
            // Retrieve boxes and scores
            std::vector<Rect2i> face_boxes;
            std::vector<float> face_scores;
            for (Face f: faces)
            {
                face_boxes.push_back(f.box_tlwh);
                face_scores.push_back(f.score);
            }

            std::vector<int> keep_idx;
            dnn::NMSBoxes(face_boxes, face_scores, score_thresh, nms_thresh, keep_idx, 1.f, top_k);

            // Get results
            std::vector<Face> nms_faces;
            for (int idx: keep_idx) 
            {
                nms_faces.push_back(faces[idx]);
            }
            return nms_faces;
        }
        else
        {
            return faces;
        }
    }
} // namespace dnn_face
} // namespace cv