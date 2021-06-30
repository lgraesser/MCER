# run instructions:
# build image: docker build -t imcap:v1.0.0 .
# start container: docker run -u 1000:1000 --mount type=bind,source="$(pwd)",target=/image-captioning --rm -it imcap:v1.0.0
# start container with GPU: docker run -u 1000:1000 --mount type=bind,source="$(pwd)",target=/image-captioning --rm --gpus all -it imcap:v1.0.0
# start container with GPU and port forwarding for jupyter: docker run -u 1000:1000 --mount type=bind,source="$(pwd)",target=/image-captioning --rm --gpus all -p 8181:8181 -it imcap:v1.0.0
# list image: docker images -a

FROM tensorflow/tensorflow:latest-gpu-jupyter

LABEL maintainer="lhgraesser@gmail.com"
LABEL website="https://github.com/lgraesser/image-captioning"

SHELL ["/bin/bash", "-c"]

# Install the additional dependencies
# tensorflow, numpy, and jupyter are included in the parent image
RUN pip install scikit-learn && \
    pip install matplotlib && \
    pip install pillow && \
    pip install absl-py

# create and set the directory to bind mount the source code and data to
RUN mkdir -p /image-captioning

WORKDIR /image-captioning

CMD ["/bin/bash"]
