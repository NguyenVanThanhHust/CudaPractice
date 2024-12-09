FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

# Install some basic utilities
RUN apt-get update --fix-missing && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    checkinstall \
    locales \
    lsb-release \
    mesa-utils \
    subversion \
    terminator \
    xterm \
    wget \
    htop \
    libssl-dev \
    build-essential \
    dbus-x11 \
    software-properties-common \
    gdb valgrind \
    libeigen3-dev \
    libboost-all-dev \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libglfw3-dev libgl-dev libglu-dev

# Install python3.10
WORKDIR /opt/
RUN apt update && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install -y python3.10-full
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py 

WORKDIR /opt/
RUN python3.10 -m pip install cmake

RUN wget --no-check-certificate -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.9.0.zip
RUN apt update && apt install unzip -y
RUN unzip opencv.zip

WORKDIR /opt/opencv-4.9.0/build/
RUN cmake .. -D BUILD_opencv_java=OFF \
      -D WITH_EIGEN=ON \
      -D BUILD_opencv_python=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D CMAKE_BUILD_TYPE=RELEASE \
        .. 
RUN make -j16 && make install && ldconfig 

RUN echo "alias ..='cd ..'" >> ~/.bashrc
RUN echo "alias ...='cd .. && cd ..'" >> ~/.bashrc
RUN echo "alias py=/usr/bin/python3" >> ~/.bashrc

WORKDIR /opt/
COPY go1.23.2.linux-amd64.tar.gz /opt/
RUN rm -rf /usr/local/go && tar -C /usr/local -xzf go1.23.2.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin
RUN go version

RUN apt update && apt install fish -y

WORKDIR /workspace