# C++ implementation of the BMVC2023 paper [Mobile Vision Transformer-based Visual Object Tracking](https://papers.bmvc2023.org/0800.pdf) and WACV2024 paper [Separable Self and Mixed Attention Transformers for Efficient Object Tracking](https://openaccess.thecvf.com/content/WACV2024/papers/Gopal_Separable_Self_and_Mixed_Attention_Transformers_for_Efficient_Object_Tracking_WACV_2024_paper.pdf)

The original Pytorch implementations is available in these links: [MVT](https://github.com/goutamyg/MVT) and [SMAT](https://github.com/goutamyg/SMAT).

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
