# Setup the base image for opencv
FROM opencvcourses/opencv-docker:latest

RUN apt-get update && apt-get -qq -y install wget

# Activate venv
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Setup DeepSparse Python API
RUN pip3 install --no-cache-dir --upgrade deepsparse

# Setup DeepSparse C++ API
RUN wget https://github.com/neuralmagic/deepsparse/releases/download/v1.2.0/deepsparse_api_demo.tar.gz -o /home/deepsparse_api_demo.tar.gz
RUN tar xfv /home/deepsparse_api_demo.tar.gz.1