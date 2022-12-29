cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_BUILD_TYPE "")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(yolov5 VERSION 0.1.0)

find_package(OpenCV REQUIRED)
find_package( Threads )

add_executable(main yolov5.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

include_directories(/home/deepsparse_api_demo/avx2_deepsparse_api_demo/include)
target_link_libraries(main 
    /home/deepsparse_api_demo/avx2_deepsparse_api_demo/bin/avx2/libdeepsparse.so 
    /home/deepsparse_api_demo/avx2_deepsparse_api_demo/bin/avx2/libonnxruntime.so.1.12.0)
