# C++ implementation of the BMVC2023 paper [Mobile Vision Transformer-based Visual Object Tracking](https://papers.bmvc2023.org/0800.pdf)

The original Pytorch implementation is available [here](https://github.com/goutamyg/MVT).

## Why C++ implementation? 
* Improve the model inference speed (*fps*) 
* Single vs Multi-thread throughput analysis
* Subsequent deployment on Andorid devices using JNI

## Supported Frameworks
* [MNN](https://github.com/alibaba/MNN)
* [ONNX-Runtime](https://github.com/microsoft/onnxruntime)

## To-do
* [TF-Lite](https://www.tensorflow.org/lite/guide)
* [NCNN](https://github.com/Tencent/ncnn)
