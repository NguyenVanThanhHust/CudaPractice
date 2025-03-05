# CudaPratice
Pratice Cuda Examples from NVIDIA

## Installation
Builde dev docker
```
docker build -f dockers/dev.Dockerfile -t cuda_dev_img:12.3.2 .
```
Build docker container
```
docker run --rm --name cuda_dev_ctn -it --gpus all --volume="$PWD:/workspace" -w /workspace/ cuda_dev_img:12.3.2 /bin/bash
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
- [ ] bandwidthTest
- [ ] [Connected componented on cuda](https://github.com/FolkeV/CUDA_CCL/tree/master) [ori](https://github.com/DanielPlayne/playne-equivalence-algorithm)
- [ ] [Connected componented 3d on cuda](https://github.com/seung-lab/connected-components-3d)

## Reference
Performance measurement: https://leimao.github.io/blog/Docker-Nsight-Compute/
Get CUDA architecture in CMake: https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures 