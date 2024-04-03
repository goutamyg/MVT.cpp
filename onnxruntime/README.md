# C++ implementation of [MVT](https://papers.bmvc2023.org/0800.pdf) and [SMAT](https://openaccess.thecvf.com/content/WACV2024/papers/Gopal_Separable_Self_and_Mixed_Attention_Transformers_for_Efficient_Object_Tracking_WACV_2024_paper.pdf) trackers using [ONNX-Runtime](https://github.com/microsoft/onnxruntime)

## Dependencies
* [OpenCV](https://github.com/opencv/opencv) 4.8.0
* [ONNX-Runtime-1.12.1](https://github.com/microsoft/onnxruntime) (Related header files and the *libonnxruntime.so* are already presented in the `third_party/` folder)
* Download the pretrained models from these links: [MVT](https://drive.google.com/file/d/15dI9j7UQc35pcWjD0133eRzLh0P_fRvx/view?usp=drive_link), [SMAT](https://drive.google.com/file/d/10K1dCgfnSyC-Y3ShMZymdWL1-ubDC3nl/view?usp=sharing)
* Place the downloaded tracker model under `model/` folder

## Build
```
$ mkdir build && cd build
$ cmake ..
$ make
```

## Run
```
$ cd build
$ ./mvt_demo [model_path] [video_path]
```
For example, 
```
./mvt_demo ../model/MobileViT_Track_ep0300.onnx ../data/input/
```

## Single vs Multi-thread *fps* evaluation

## Python demo
Run the onnxruntime-based python demo file as,
```
python3 ./python3 mvt_onnxruntime_demo.py
```
Make sure you install the required libraries.

## Some lessons learned
* [Occasional *NaN* results](https://github.com/microsoft/onnxruntime/issues/19851)

## To-Do
* GPU-based inference

## Acknowledgements
* [ONNX Runtime: A cross-platform inference and training machine-learning accelerator](https://github.com/microsoft/onnxruntime)
