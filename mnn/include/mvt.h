//
// Created by zxiong on 23-3-31.
//

#ifndef MVT_H
#define MVT_H

#include <vector> 
#include <map>
#include <memory>

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Matrix.h>
#include <MNN/Tensor.hpp>

#include "opencv2/opencv.hpp"

struct BBox {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct MVT_Output {
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

    void init(const cv::Mat &img, MVT_Output bbox);
    
    const MVT_Output& track(const cv::Mat &img);
    
    // state  dynamic
    BBox state;
    
    // config static
    Config cfg; 

private:

    void map_box_back(BBox &pred_box, float resize_factor);

    void clip_box(BBox &box, int height, int wight, int margin);

    void cal_bbox(MNN::Tensor &scores_tensor, MNN::Tensor &offsets_tensor, MNN::Tensor &sizes_tensor, BBox &pred_box, float &max_score, float resize_factor);

    void sample_target(const cv::Mat &im, cv::Mat &croped, BBox target_bb, float search_area_factor, int output_sz, float &resize_factor);

    const float means[3]  = {0.0, 0.0, 0.0}; // BGR
    const float norms[3] = {1.0/255, 1.0/255, 1.0/255}; // BGR
    
    std::unique_ptr<MNN::Interpreter> mnn_net = nullptr;
    MNN::Session* session = nullptr;

    MNN::Tensor* x = nullptr; // input-1 
    MNN::Tensor* z = nullptr; // input-2

    MNN::ScheduleConfig net_config;

    cv::Mat z_patch;

    MNN::CV::ImageProcess::Config config;

    std::unique_ptr<MNN::CV::ImageProcess> preprocess;

    MVT_Output object_box;
};

#endif 
