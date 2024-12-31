# CudaPratice
Pratice Cuda Examples from NVIDIA

## Installation
Builde dev docker
```
docker build -f dockers/dev.Dockerfile --tag=cuda_dev_img:12.3.2 .
```
Build docker container
```
docker run --rm --name cuda_dev_ctn -it --gpus all --volume="$PWD:/workspace" cuda_dev_img:12.3.2 /bin/bash
```
Start docker container 
```
docker start cuda_dev_ctn && docker exec -it cuda_dev_ctn /bin/bash
```

Build performance benchmark docker
```
docker build -f dockers/nsight-compute.Dockerfile --tag=cuda_benchmark_img:12.3.2 .
```

## TODO
- [ ] Batch Matrix multiplication template
- [ ] Python binding template
- [ ] Golang binding template

## Reference
Performance measurement: https://leimao.github.io/blog/Docker-Nsight-Compute/
Get CUDA architecture in CMake: https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures 