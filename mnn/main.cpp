#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>

#include "mvt.h"

void track(MVT *tracker, const char *video_path)
{
    std::string folderPath = video_path;

    // Open the folder
    std::string folderGlob = folderPath + "*.jpg"; // Change the file extension if necessary
    std::vector<std::string> imageFiles;
    cv::glob(folderGlob, imageFiles);
    
    // Read the annotaiton file
    std::ifstream inputFile(folderPath + "../groundtruth.txt");
    std::string groundtruth_anno;
    if (std::getline(inputFile, groundtruth_anno)) {
        std::cout << "Initial bounding-box annotation: " << groundtruth_anno << std::endl;
    } else {
        std::cerr << "Failed to read the annotation file" << std::endl;
    }

    // Initial bounding box
    std::string bboxString(groundtruth_anno);
    float x, y, w, h;
    std::istringstream iss(bboxString);
    char delimiter;
    if (!(iss >> x >> delimiter >> y >> delimiter >> w >> delimiter >> h)) {
        std::cerr << "Error: Invalid input format" << std::endl;
    }    
    cv::Rect trackWindow(x, y, w, h);

    int frame_idx = 0;
    bool isFirstImage = true;
    for (const auto& imagePath : imageFiles) {
        // Read the image
        cv::Mat frame = cv::imread(imagePath);

        if (isFirstImage){

            std::cout << "Start track init ..." << std::endl;
            std::cout << "==========================" << std::endl;
            MVT_Output bbox;
            bbox.box.x0 = trackWindow.x;
            bbox.box.x1 = trackWindow.x+trackWindow.width;
            bbox.box.y0 = trackWindow.y;
            bbox.box.y1 = trackWindow.y+trackWindow.height;
            tracker->init(frame, bbox);
            std::cout << "==========================" << std::endl;
            std::cout << "Init done!" << std::endl;
            std::cout << std::endl;
            isFirstImage = false;
        }
        else{
            // Start timer
            double t = (double)cv::getTickCount();

            // Update tracker.
            MVT_Output bbox = tracker->track(frame);

            // Calculate Frames per second (FPS)
            double inference_time = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

            // Result to rect.
            cv::Rect rect;
            rect.x = bbox.box.x0;
            rect.y = bbox.box.y0;
            rect.width = int(bbox.box.x1 - bbox.box.x0);
            rect.height = int(bbox.box.y1 - bbox.box.y0);

            std::cout << "[x0, y0, w, h]: [" << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << "]" << std::endl;
            std::cout << "target classification score: " << bbox.score << std::endl;

            // Boundary judgment.
            cv::Mat track_window;
            if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= frame.cols && 0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= frame.rows)
            {
                cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
            }

            // Save the tracker output images
            std::string dst_folderPath = "../data/output/";
            std::string imagePath = dst_folderPath + std::to_string(frame_idx) + ".jpg"; // Adjust the file extension if necessary
            cv::imwrite(imagePath, frame);

            // Display per-frame inference time 
            std::cout << "Inference time: " << inference_time << " seconds" << std::endl;
            std::cout << "==========================" << std::endl;
            std::cout << std::endl;
        }
        frame_idx += 1;
    }
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [modelpath] [videopath]\n", argv[0]);
        return -1;
    }

    // Get model path.
    const char *model_path = argv[1];

    // Get video path.
    const char *video_path = argv[2]; 

    // Build tracker.
    MVT* MVT_tracker;
    MVT_tracker = new MVT(model_path);
    track(MVT_tracker, video_path);

    return 0;
}
