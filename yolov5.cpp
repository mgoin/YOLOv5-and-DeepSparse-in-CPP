// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono>
#include <assert.h>
#include <thread>
#include <string>

#include "libdeepsparse/engine.hpp"

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants.
const int INPUT_WIDTH_INT = 640;
const int INPUT_HEIGHT_INT = 640;
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
auto now() { return std::chrono::high_resolution_clock::now(); }
double milliseconds_between(time_point const &start,
                            time_point const &end)
{
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return static_cast<double>(duration) / 1000;
}

double seconds_between(time_point const &start,
                            time_point const &end)
{
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return static_cast<double>(duration) / 1000000;
}

void print_element_type(const deepsparse::element_type_t elt_type) {
    switch(elt_type) {
        case deepsparse::element_type_t::invalid:
            std::cout << "invalid\n";
            return;
        case deepsparse::element_type_t::boolean:
            std::cout << "boolean\n";
            return;
        case deepsparse::element_type_t::int8:
            std::cout << "int8\n";
            return;
        case deepsparse::element_type_t::int16:
            std::cout << "int16\n";
            return;
        case deepsparse::element_type_t::int32:
            std::cout << "int32\n";
            return;
        case deepsparse::element_type_t::int64:
            std::cout << "int64\n";
            return;
        case deepsparse::element_type_t::uint8:
            std::cout << "uint8\n";
            return;
        case deepsparse::element_type_t::float32:
            std::cout << "float32\n";
            return;
        case deepsparse::element_type_t::float64:
            std::cout << "float64\n";
            return;
        default:
            std::cout << "UNRECOGNIZED\n";
            return;
    }
}

// Draw the predicted bounding box.
void draw_label(Mat &input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


// quantized yolov5 expects 0-255 uint8 inputs
// non-quantized yolov5 expects 0-1. fp32 inputs
vector<Mat> pre_process_deepsparse(Mat &input_image, deepsparse::engine_t &engine, bool is_quantized)
{   
    // Convert to blob.
    Mat blob;
    std::vector<deepsparse::tensor_t> inputs;

    // if model is not quantized, expects inputs from 0->1 as a float
    if (!is_quantized) {
        blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT_INT), Scalar(), true, false);
        assert(blob.isContinuous());

        // creates tensor with the blob raw data        
        deepsparse::tensor_t input(deepsparse::element_type_t::float32, deepsparse::dimensions_t({1, 3, INPUT_WIDTH, INPUT_HEIGHT}), blob.data, [](void *p) {});
        inputs.push_back(input);
    
    // if model is quantized, expects inputs from 0-255 as uint8_t
    } else {
        blobFromImage(input_image, blob, 1., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
        assert(blob.isContinuous());
        
        // HACK: round down floats to uint8         
        float* float_data = reinterpret_cast<float*>(blob.data);
        uint8_t uint8_data[3 * INPUT_HEIGHT_INT * INPUT_HEIGHT_INT];
        for (int i = 0; i < 3 * INPUT_HEIGHT_INT * INPUT_HEIGHT_INT; i++) {
            assert(float_data[i] <= 255.0 && float_data[i] >= 0.);  // confirms pixel values that fit in uint8
            uint8_data[i] = static_cast<uint8_t>(float_data[i]);    // cast to uint8, rounding down
        }

        // creates tensor with the uint8 version of the blob data
        deepsparse::tensor_t input(deepsparse::element_type_t::uint8, deepsparse::dimensions_t({1, 3, INPUT_WIDTH, INPUT_HEIGHT}), uint8_data, [](void *p) {});
        inputs.push_back(input);
    }

    // Forward propagate.
    std::vector<deepsparse::tensor_t> outputs = engine.execute(inputs);

    vector<Mat> cv_outputs;
    for (deepsparse::tensor_t &t : outputs)
    {
        std::vector<int> dims(t.dims().begin(), t.dims().end());
        // HACK: read output type from deepsparse
        cv_outputs.push_back(cv::Mat(dims, blob.type(), t.data<void>()));
    }

    return cv_outputs;
}

Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name)
{   
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    
    return input_image;
}

// worker thread for latency testing
void test_latency(deepsparse::engine_t* e_ptr, std::vector<double>* latencies, deepsparse::tensors_t tensors, size_t iters) 
{
    for (size_t i = 0; i < iters; i++) {
        auto start_time = now();
            auto resp = e_ptr->execute(tensors);
        auto end_time = now();
        latencies->push_back(milliseconds_between(start_time, end_time));
    }
}

void run_iterations(deepsparse::engine_t &engine, deepsparse::tensors_t &tensors, size_t iters) {
    for (size_t i = 0; i < iters; i++) {
        auto resp = engine.execute(tensors);
    }
}

int main(int argc, char* argv[])
{    
    if (argc < 4) {
        std::cout << "main [image_path] [onnx_model_path] [batch_size] [iterations]\n";
        return -1;
    }
    auto img_path = argv[1];
    auto onnx_path = argv[2];
    size_t batch_size = static_cast<size_t>(std::stoi(argv[3]));
    size_t num_iters = static_cast<size_t>(std::stoi(argv[4]));

    // Load class list.
    vector<string> class_list;
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    {
        std::cout << "Starting Latency Test With " << num_iters << " Iterations\n";

        std::cout << "Loading DeepSparse With Batch Size " << batch_size << "\n";
        deepsparse::engine_config_t config{onnx_path, batch_size};
        auto engine = deepsparse::engine_t(config);
        auto random_tensors = deepsparse::generate_random_inputs(engine);

        std::cout << "Running Throughput Testing \n";
        auto start_time = now();
        run_iterations(engine, random_tensors, num_iters);
        auto end_time = now();

        size_t total_iters = batch_size * num_iters;
        double throughput = static_cast<double>(total_iters) / seconds_between(start_time, end_time);
        std::cout << "Total Seconds = " << seconds_between(start_time, end_time) << "\n";
        std::cout << "Num Iters = " << total_iters << "\n";
        std::cout << "Throughput (items/sec): " << throughput << "\n\n";

        // create engine with batch 1 to run example pipeline
        std::cout << "Running Example Pipeline\n";
        std::cout << "Loading DeepSparse With Batch Size 1\n";
        deepsparse::engine_config_t config_batch_1{onnx_path};
        auto engine_batch_1 = deepsparse::engine_t(config_batch_1);

        // if model is quantized, pass image as 0-255 uint8
        // if model is not quantized, pass image as 0.-1. floats
        auto input_type = engine_batch_1.input_element_type(0);
        bool is_quantized = false;
        if (*input_type == deepsparse::element_type_t::uint8) {
            is_quantized = true;
        } else {
            assert(*input_type == deepsparse::element_type_t::float32);
        }

        // Load image.
        Mat frame;
        frame = imread(img_path);

        // Run DeepSparse
        vector<Mat> detections = pre_process_deepsparse(frame, engine_batch_1, is_quantized);
        Mat img = post_process(frame, detections, class_list);
    
        imwrite("output_cv-deepsparse.jpg", img);
        std::cout << "Saved DeepSparse output to output_cv-deepsparse.jpg\n";
    }
}
