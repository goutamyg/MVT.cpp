#include <cstdlib>
#include <string>
#include <cmath>

#include "mvt.h"

#define TIME
#ifdef TIME
#include <sys/time.h>
#endif

#ifdef TIME
    struct timeval tv;
    uint64_t time_last;
    double time_ms;
#endif

static std::vector<float> hann(int sz) {
    std::vector<float> hann1d(sz);
    std::vector<float> hann2d(sz * sz);
    for (int i = 1; i < sz + 1; i++) {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (sz+1) );
        hann1d[i-1] = w;
    }
    for (int i = 0; i < sz; i++) {
        for (int j=0; j < sz; j++) {
            hann2d[i*sz + j] = hann1d[i] * hann1d[j];
        }
    }
   return hann2d;
}

size_t vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

bool inRange(const float* ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (ptr[i] < 0.0f or ptr[i] > 1.0f) {
            return false; // If any element is not between 0.0 and 1.0, return false immediately
        }
    }
    return true;
}

// function to round-off to the nearest integer (and next higher integer in case of a tie)
int customRound(float x) {
    return static_cast<int>(x < 0.0f ? std::ceil(x) : std::floor(x));
}

MVT::MVT(const char *model_path) {

    // Create ONNX Runtime environment with warning logging level
    this->env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    
    // Create session options
    sessionOptions = Ort::SessionOptions();

    // Set the number of intra-op threads (parallelism within a single operator)
    sessionOptions.SetIntraOpNumThreads(8); // Set to desired number of threads

    // Set the number of inter-op threads (parallelism across operators)
    sessionOptions.SetInterOpNumThreads(8); // Set to desired number of threads

    // Create ONNX Runtime session with the specified model path and session options
    this->session = Ort::Session(env, model_path, sessionOptions);

    // Create allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input tensor shape for target template
    Ort::TypeInfo inputTypeInfo_z = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape_z = inputTypeInfo_z.GetTensorTypeAndShapeInfo().GetShape();

    // Get input tensor shape for search region
    Ort::TypeInfo inputTypeInfo_x = session.GetInputTypeInfo(1);
    std::vector<int64_t> inputTensorShape_x = inputTypeInfo_x.GetTensorTypeAndShapeInfo().GetShape();

    // Print input tensor shapes for target template and search region
    for (auto shape : inputTensorShape_z)
        std::cout << "Input shape for target template: " << shape << std::endl;

    for (auto shape : inputTensorShape_x)
        std::cout << "Input shape for search region: " << shape << std::endl;

    // Get input and output names from the session
    this->inputNames.push_back(session.GetInputName(0, allocator)); // not supported in 1.16.1, but works under 1.12.1
    this->inputNames.push_back(session.GetInputName(1, allocator)); // not supported in 1.16.1, but works under 1.12.1

    this->outputNames.push_back(session.GetOutputName(0, allocator)); // not supported in 1.16.1, but works under 1.12.1
    this->outputNames.push_back(session.GetOutputName(1, allocator)); // not supported in 1.16.1, but works under 1.12.1
    this->outputNames.push_back(session.GetOutputName(2, allocator)); // not supported in 1.16.1, but works under 1.12.1
    this->outputNames.push_back(session.GetOutputName(3, allocator)); // not supported in 1.16.1, but works under 1.12.1

    // Generate hann2d window
    this->cfg.window = hann(this->cfg.feat_sz);
}

void MVT::init(const cv::Mat &img, MVT_output bbox)
{
    // Get subwindow.
    cv::Mat z_patch;
    float resize_factor = 1.f;
    this->sample_target(img, z_patch, bbox.box, this->cfg.template_factor, this->cfg.template_size, resize_factor);
    this->z_patch = z_patch;
    this->state = bbox.box;
}


void MVT::preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize {floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

void MVT::prepareInputTensors(cv::Mat &image_z, cv::Mat &image_x, 
                            std::vector<float>& inputTensorValues_Z, std::vector<float>& inputTensorValues_X, 
                            std::vector<int64_t>& inputTensorShape_Z, std::vector<int64_t>& inputTensorShape_X, 
                            std::vector<Ort::Value>& inputTensors)
{

    size_t inputTensorSize_Z = vectorProduct(inputTensorShape_Z);
    size_t inputTensorSize_X = vectorProduct(inputTensorShape_X);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, 
                                                    inputTensorValues_Z.data(), 
                                                    4*inputTensorSize_Z, 
                                                    inputTensorShape_Z.data(), 
                                                    inputTensorShape_Z.size(), 
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

    inputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, 
                                                    inputTensorValues_X.data(), 
                                                    4*inputTensorSize_X,
                                                    inputTensorShape_X.data(), 
                                                    inputTensorShape_X.size(), 
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
}

const MVT_output& MVT::track(const cv::Mat &img)
{
    // Get subwindow.
    cv::Mat x_patch;
    float resize_factor = 1.f;
    this->sample_target(img, x_patch, this->state, this->cfg.search_factor, this->cfg.search_size, resize_factor);

    // Pre-process.
    float* blob_x = nullptr;
    std::vector<int64_t> inputTensorShape_X {1, 3, -1, -1};
    this->preprocessing(x_patch, blob_x, inputTensorShape_X);

    // check the input-X is in the range 0-1
    size_t inputTensorSize_X = vectorProduct(inputTensorShape_X);
    assert(inRange(blob_x, inputTensorSize_X) == true);

    float* blob_z = nullptr;
    std::vector<int64_t> inputTensorShape_Z {1, 3, -1, -1};
    this->preprocessing(this->z_patch, blob_z, inputTensorShape_Z);

    size_t inputTensorSize_Z = vectorProduct(inputTensorShape_Z);
    assert(inRange(blob_z, inputTensorSize_Z) == true);

    std::vector<float> inputTensorValues_Z(blob_z, blob_z + inputTensorSize_Z);
    std::vector<float> inputTensorValues_X(blob_x, blob_x + inputTensorSize_X);

    std::vector<Ort::Value> ortInputs;
    this->prepareInputTensors(this->z_patch, x_patch, inputTensorValues_Z, inputTensorValues_X, inputTensorShape_Z, inputTensorShape_X, ortInputs);
    
    // Run inference
    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                         this->inputNames.data(),
                                                         ortInputs.data(),
                                                         this->inputNames.size(),
                                                         this->outputNames.data(),
                                                         this->outputNames.size());

    delete[] blob_z;
    delete[] blob_x;

    // Fetch output tensors
    std::vector<std::vector<float>> outputData;
    for (auto& outputTensor : outputTensors) {
        // Get data type and shape of the output tensor
        Ort::TypeInfo outputTypeInfo = outputTensor.GetTypeInfo();
        auto tensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = tensorInfo.GetShape();

        // Get data from the output tensor
        auto outputDataPtr = outputTensor.GetTensorMutableData<float>();
        std::vector<float> output(outputDataPtr, outputDataPtr + tensorInfo.GetElementCount());

        /*
        // Print output shape
        std::cout << "Output Shape: [";
        for (size_t j = 0; j < outputShape.size(); ++j) {
            std::cout << outputShape[j];
            if (j != outputShape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        
        // Get output tensor values
        size_t numElements = tensorInfo.GetElementCount();
        std::cout << "Output Values:" << std::endl;
        for (size_t j = 0; j < numElements; ++j) {
            std::cout << outputDataPtr[j] << " ";
        }
        std::cout << " " << std::endl;
        */

        // Store output data for further processing
        outputData.push_back(output);
    }
    auto scores  = outputData[1];
    auto sizes   = outputData[2];
    auto offsets = outputData[3];

    // Post-process the tracker output
    BBox pred_box;
    float max_score;
    this->cal_bbox(scores, offsets, sizes, pred_box, max_score, resize_factor);
    this->map_box_back(pred_box, resize_factor);
    this->clip_box(pred_box, img.rows, img.cols, 10);
    
    object_box.box = pred_box;
    object_box.score = max_score;

    this->state = object_box.box;

    return object_box;
}

void MVT::cal_bbox(std::vector<float> &scores_tensor, std::vector<float> &offsets_tensor, std::vector<float> &sizes_tensor, BBox &pred_box, float &max_score, float resize_factor) {
    // Add hann window, get max value and index.
    auto scores_ptr = scores_tensor;
    float max_value = this->cfg.window[0] * scores_ptr[0];
    int max_idx_y = 0; int max_idx_x = 0; int max_idx = 0;
    float tmp_score = 0.f;
    for (int i = 0; i < scores_tensor.size(); i++) {
        tmp_score = this->cfg.window[i] * scores_ptr[i];
        if (tmp_score > max_value) {
            max_idx = i;
            max_value = tmp_score;
        }
    }

    max_idx_y = max_idx / this->cfg.feat_sz;
    max_idx_x = max_idx % this->cfg.feat_sz;

    auto sizes_ptr = sizes_tensor;
    auto offsets_ptr = offsets_tensor;

    float cx = (max_idx_x + offsets_ptr[max_idx_y * this->cfg.feat_sz + max_idx_x]) * 1.f / this->cfg.feat_sz;
    float cy = (max_idx_y + offsets_ptr[this->cfg.feat_sz * this->cfg.feat_sz + max_idx_y * this->cfg.feat_sz + max_idx_x]) *1.f / this->cfg.feat_sz;

    float w = sizes_ptr[max_idx_y * this->cfg.feat_sz + max_idx_x];
    float h = sizes_ptr[this->cfg.feat_sz * this->cfg.feat_sz + max_idx_y * this->cfg.feat_sz + max_idx_x];
  
    cx = cx * this->cfg.search_size / resize_factor;
    cy = cy * this->cfg.search_size / resize_factor;
    w = w * this->cfg.search_size / resize_factor;
    h = h * this->cfg.search_size / resize_factor;
    
    pred_box.x0 = cx - 0.5 * w;
    pred_box.y0 = cy - 0.5 * h;
    pred_box.x1 = pred_box.x0 + w;
    pred_box.y1 = pred_box.y0 + h;    

    max_score = max_value;
}

void MVT::map_box_back(BBox &pred_box, float resize_factor) {
    float cx_prev = this->state.x0 + 0.5 * (this->state.x1 - this->state.x0);
    float cy_prev = this->state.y0 + 0.5 * (this->state.y1 - this->state.y0);

    float half_side = 0.5 * this->cfg.search_size / resize_factor;

    float w = pred_box.x1 - pred_box.x0;
    float h = pred_box.y1 - pred_box.y0;
    float cx = pred_box.x0 + 0.5 * w;
    float cy = pred_box.y0 + 0.5 * h;

    float cx_real = cx + (cx_prev - half_side);
    float cy_real = cy + (cy_prev - half_side);

    pred_box.x0 = cx_real - 0.5 * w;
    pred_box.y0 = cy_real - 0.5 * h;
    pred_box.x1 = cx_real + 0.5 * w;
    pred_box.y1 = cy_real + 0.5 * h;
}

void MVT::clip_box(BBox &box, int height, int wight, int margin) {
    box.x0 = std::min(std::max(0, int(box.x0)), wight - margin);
    box.y0 = std::min(std::max(0, int(box.y0)), height - margin);
    box.x1 = std::min(std::max(margin, int(box.x1)), wight);
    box.y1 = std::min(std::max(margin, int(box.y1)), height);
}

void MVT::sample_target(const cv::Mat &im, cv::Mat &cropped, BBox target_bb, float search_area_factor, int output_sz, float &resize_factor) {
    /* Extracts a square crop centrered at target_bb box, of are search_area_factor^2 times target_bb area
    args:
        im: Img image
        target_bb - target box [x0, y0, x1, y1]
        search_area_factor - Ratio of crop size to target size
        output_sz - Size

    */
    int x = target_bb.x0;
    int y = target_bb.y0;
    int w = target_bb.x1 - target_bb.x0;
    int h = target_bb.y1 - target_bb.y0;
    int crop_sz = std::ceil(std::sqrt(w*h) * search_area_factor);

    float cx = x + 0.5 * w;
    float cy = y + 0.5 * h;
    int x1 = customRound(cx - crop_sz * 0.5);
    int y1 = customRound(cy - crop_sz * 0.5);

    int x2 = x1 + crop_sz;
    int y2 = y1 + crop_sz;

    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(x2 - im.cols +1, 0);

    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(y2- im.rows + 1, 0);

    // Crop target
    cv::Rect roi_rect(x1+x1_pad, y1+y1_pad, (x2-x2_pad)-(x1+x1_pad), (y2-y2_pad)-(y1+y1_pad));
    cv::Mat roi = im(roi_rect);

    // Pad
    cv::copyMakeBorder(roi, cropped, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);

    // Resize
    cv::resize(cropped, cropped, cv::Size(output_sz, output_sz));

    resize_factor = output_sz * 1.f / crop_sz;
}
