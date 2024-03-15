#ifndef MVT_H
#define MVT_H

#include <vector> 
#include <map>
#include <memory>

#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>

struct BBox {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct MVT_output {
    BBox box;
    float score;
};

struct Config {
    std::vector<float> window;

    float template_factor = 2.0;
    float search_factor = 4.0;
    float template_size = 128;
    float search_size = 256;
    float stride = 16;
    int feat_sz = 16;
};

class MVT {

    public: 
        
        MVT(const char *model_path);
        
        ~MVT(); 

        void init(const cv::Mat &img, MVT_output bbox);
        
        const MVT_output& track(const cv::Mat &img);
        
        // state  dynamic
        BBox state;
        
        // config static
        Config cfg; 

    private:

        Ort::Env env{nullptr};
        Ort::SessionOptions sessionOptions{nullptr};
        Ort::Session session{nullptr};
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        void sample_target(const cv::Mat &im, cv::Mat &croped, BBox target_bb, float search_area_factor, int output_sz, float &resize_factor);

        void preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape);

        void prepareInputTensors(cv::Mat &image_z, cv::Mat &image_x, 
                            std::vector<float>& inputTensorValues_Z, std::vector<float>& inputTensorValues_X, 
                            std::vector<int64_t>& inputTensorShape_Z, std::vector<int64_t>& inputTensorShape_X, 
                            std::vector<Ort::Value>& inputTensors);

        void cal_bbox(std::vector<float> &scores_tensor, std::vector<float> &offsets_tensor, std::vector<float> &sizes_tensor, 
                            BBox &pred_box, float &max_score, float resize_factor);

        void clip_box(BBox &box, int height, int wight, int margin);

        void map_box_back(BBox &pred_box, float resize_factor);

        // const float means[3]  = {0.0, 0.0, 0.0}; // BGR
        // const float norms[3] = {1.0/255, 1.0/255, 1.0/255}; // BGR
        
        cv::Mat z_patch;

        MVT_output object_box;
};

#endif 
