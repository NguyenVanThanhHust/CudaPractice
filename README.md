# CudaPratice
Pratice Cuda Examples from NVIDIA

## Installation
Builde dev docker
```
docker build -f dockers/dev.Dockerfile -t cuda_dev_img:12.3.2 .
```
Build docker container
```
docker run --rm --name cuda_dev_ctn -it --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all --volume="$PWD:/workspace" -w /workspace/ cuda_dev_img:12.3.2 fish
```

Build performance benchmark docker
```
docker build -f dockers/nsight-compute.Dockerfile --tag=cuda_benchmark_img:12.3.2 .
```
## How to run

## TODO
- [ ] bandwidthTest
- [ ] [Connected componented on cuda](https://github.com/FolkeV/CUDA_CCL/tree/master) [ori](https://github.com/DanielPlayne/playne-equivalence-algorithm)
- [ ] [Connected componented 3d on cuda](https://github.com/seung-lab/connected-components-3d)

## Reference
Performance measurement: https://leimao.github.io/blog/Docker-Nsight-Compute/
Get CUDA architecture in CMake: https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures

Other CUDA with CMake
https://github.com/GeneTinderholm/cuda-example/tree/main