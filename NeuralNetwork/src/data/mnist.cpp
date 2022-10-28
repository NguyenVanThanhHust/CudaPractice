#include "mnist.hpp"


void read_data(std::string data_path, thrust::host_vector<thrust::host_vector<float>> &data)
{
    std::vector<std::string> image_paths;
    for (const auto & entry: std::filesystem::directory_iterator(data_path))
    {
        image_paths.push_back(entry.path());
    }

    int num_image=0;
    for (auto image_path: image_paths)
    {
        thrust::host_vector<float> single_image_data;
        cv::Mat img = cv::imread(image_path);
        int width = img.cols;
        int height = img.rows;
        int stride = img.step;
        for (int row_index = 0; row_index < height; row_index++)
        {
            for (int col_index = 0; col_index < width; col_index++)
            {
                float value = img.at<float>(row_index, col_index);
                single_image_data.push_back(value);
            }
        }
        data.push_back(single_image_data);
        num_image += 1;
    }
    cout<<"Number of images: "<<num_image<<endl;
}