# Object Detection in C++ using YOLOv5 and DeepSparse + OpenCV

*Note: This demo Uses the AVX2 binary. If you have an AVX512 machine, you need to update `CMakeLists.txt` to use the AVX512 binary.*

## How to run the demo

This demo runs throughput testing and then uses uses OpenCV to perform the YOLOv5 pre-processing and post-processing around DeepSparse. 

* Build and run the docker image
```
docker build -t yolo-opencv .
docker run -v $(pwd):/home/demo -it yolo-opencv
```

* Build and run the C++ demo
```
cd demo
mkdir build
cd build
cmake ..
cmake --build .
cd ..
```

The folder includes three versions of YOLOv5l downloaded from SparseZoo.

The command `./build/main [path_to_img] [path_to_onnx] [batch_size] [num_iterations]`
create an instance of DeepSparse with batch size `batch_size` and runs `num_iterations` forward passes to compute throughput.
Then, it creates an instance of DeepSparse with batch size 1 and runs an end-to-end YOLOv5 inference pipeline on the image provided.

* Run a pruned-quantized version of YOLOv5l
```
./build/main traffic.jpg yolov5l-pruned-quant.onnx 1 100
```

* Run a pruned version of YOLOv5l
```
./build/main traffic.jpg yolov5l-pruned.onnx 1 100
```

* Run a dense version of YOLOv5l
```
./build/main traffic.jpg yolov5l-dense.onnx 1 100
```

Example Batch 64 Pruned-Quantized (on `c4.8xlarge` which has 18 cores and AVX2 ISA):
```
root@b4d7c77b083e:/home/demo# ./build/main traffic.jpg yolov5l-pruned-quant.onnx 64 25
Starting Latency Test With 25 Iterations
Loading DeepSparse With Batch Size 64
DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. version: 1.2.0 COMMUNITY EDITION | (45d54d49) (release) (optimized) (system=avx2, binary=avx2)
Running Throughput Testing 
Total Seconds = 47.3611
Num Iters = 1600
Throughput (items/sec): 33.783

Running Example Pipeline
Loading DeepSparse With Batch Size 1
Saved DeepSparse output to output_cv-deepsparse.jpg
```

Example Batch 1 Pruned-Quantized (on `c4.8xlarge` which has 18 cores and AVX2 ISA):
```
root@b4d7c77b083e:/home/demo# ./build/main traffic.jpg yolov5l-pruned-quant.onnx 1 100
Starting Latency Test With 100 Iterations
Loading DeepSparse With Batch Size 1
DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. version: 1.2.0 COMMUNITY EDITION | (45d54d49) (release) (optimized) (system=avx2, binary=avx2)
Running Throughput Testing 
Total Seconds = 9.08185
Num Iters = 100
Throughput (items/sec): 11.011

Running Example Pipeline
Loading DeepSparse With Batch Size 1
Saved DeepSparse output to output_cv-deepsparse.jpg
```

Example Batch 64 Dense (on `c4.8xlarge` which has 18 cores and AVX2 ISA): 
```
root@b4d7c77b083e:/home/demo# ./build/main traffic.jpg yolov5l-dense.onnx 64 25
Starting Latency Test With 25 Iterations
Loading DeepSparse With Batch Size 64
DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. version: 1.2.0 COMMUNITY EDITION | (45d54d49) (release) (optimized) (system=avx2, binary=avx2)
Running Throughput Testing 
Total Seconds = 140.264
Num Iters = 1600
Throughput (items/sec): 11.4071

Running Example Pipeline
Loading DeepSparse With Batch Size 1
Saved DeepSparse output to output_cv-deepsparse.jpg
```
Example Batch 1 Dense (on `c4.8xlarge` which has 18 cores and AVX2 ISA): 
```
root@b4d7c77b083e:/home/demo# ./build/main traffic.jpg yolov5l-dense.onnx 1 100
Starting Latency Test With 100 Iterations
Loading DeepSparse With Batch Size 1
DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. version: 1.2.0 COMMUNITY EDITION | (45d54d49) (release) (optimized) (system=avx2, binary=avx2)
Running Throughput Testing 
Total Seconds = 20.7336
Num Iters = 100
Throughput (items/sec): 4.82309

Running Example Pipeline
Loading DeepSparse With Batch Size 1
Saved DeepSparse output to output_cv-deepsparse.jpg
```
To perform further benchmarking, the DeepSparse package has a benchmarking script that allows you to test out different scenarios (for example, number of streams, batch size, scheduler, etc). Install with `pip install deepsparse` (we recommend using a virtual enviornment) and running `deepsparse.benchmark --help` for usage.

**This is based off of the [Object Detection using YOLOv5 and OpenCV DNN in C++ and Python](https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/) blogpost**.

<img src="https://learnopencv.com/wp-content/uploads/2022/04/yolov5-feature-image.gif" alt="Introduction to YOLOv5 with OpenCV DNN" width="950">
