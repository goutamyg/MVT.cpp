# C++ implementation of [MVT Tracker](https://papers.bmvc2023.org/0800.pdf) using the [ONNX-Runtime](https://github.com/microsoft/onnxruntime)

## Dependencies
* [OpenCV](https://github.com/opencv/opencv) 4.8.0
* [ONNX-Runtime-1.12.1](https://github.com/microsoft/onnxruntime) (Related header files and the *libonnxruntime.so* are already presented in the `third_party/` folder)
* Download the pretrained model from [here](https://drive.google.com/file/d/15dI9j7UQc35pcWjD0133eRzLh0P_fRvx/view?usp=drive_link) and place it under `model/` folder

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


## To-Do
* Fix the occasional *NaN* results
* GPU-based inference

## Acknowledgements
* [ONNX Runtime: A cross-platform inference and training machine-learning accelerator](https://github.com/microsoft/onnxruntime)
