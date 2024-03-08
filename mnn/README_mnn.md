# MVT.cpp
C++ implementation of [Mobile Vision Transformer-based Visual Object Tracking](https://papers.bmvc2023.org/0800.pdf) using the [MNN framework](https://github.com/alibaba/MNN) by [Alibaba](https://github.com/alibaba)

## Dependencies
* Download the pretrained model from [here](https://drive.google.com/file/d/1F6iThNlFcyxfDeworOz170BhH2AU4G3h/view?usp=drive_link) under `model/` folder

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

## Single vs Multi-core evaluation


## To-Do


## Acknowledgements
* [MNN: A UNIVERSAL AND EFFICIENT INFERENCE ENGINE](https://arxiv.org/pdf/2002.12418.pdf) 
* [MNN-based implementation of OSTrack (ECCV2022)](https://github.com/Z-Xiong/OSTrack-mnn)
* [MNN: English Documenation](https://www.yuque.com/mnn/en)
