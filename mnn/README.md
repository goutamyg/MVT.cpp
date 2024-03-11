# C++ implementation of [MVT Tracker](https://papers.bmvc2023.org/0800.pdf) using the [MNN framework](https://github.com/alibaba/MNN)

## Dependencies
* [OpenCV](https://github.com/opencv/opencv) 4.8.0
* [MNN](https://github.com/alibaba/MNN) (Related header files and the *libMNN.so* are already presented in the `third_party/` folder)
* Download the pretrained model from [here](https://drive.google.com/file/d/1F6iThNlFcyxfDeworOz170BhH2AU4G3h/view?usp=drive_link) and place it under `model/` folder

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
./mvt_demo ../model/MobileViT_Track_ep0300.mnn ../data/input/
```

## Single vs Multi-thread *fps* evaluation


## To-Do
* GPU-based inference

## Acknowledgements
* [MNN: A UNIVERSAL AND EFFICIENT INFERENCE ENGINE](https://arxiv.org/pdf/2002.12418.pdf) 
* [MNN-based implementation of OSTrack (ECCV2022)](https://github.com/Z-Xiong/OSTrack-mnn)
* [MNN: English Documenation](https://www.yuque.com/mnn/en)
