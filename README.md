# CudaPratice
Pratice Cuda Examples from NVIDIA

## Installation
Builde dev docker
```
docker build -f dockers/dev.Dockerfile --tag=cuda_dev_img:12.1.1 .
```

Start docker container
```
docker run --rm --name cuda_dev_ctn -it --gpus all --volume="$PWD:/workspace" cuda_dev_img:12.1.1 /bin/bash
```

Build performance benchmark docker
```
docker build -f dockers/nsight-compute.Dockerfile --tag=cuda_benchmark_img:12.1.1 .
```

## TODO
- [ ] Convert opencv from BGR to gray format on CUDA. 

## Reference
Performance measurement: https://leimao.github.io/blog/Docker-Nsight-Compute/
Get CUDA architecture in CMake: https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures 