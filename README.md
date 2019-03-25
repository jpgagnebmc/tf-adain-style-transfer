<p>
    <a href="https://cloud.docker.com/u/deepaiorg/repository/docker/deepaiorg/tf-adain-style-transfer">
        <img src='https://img.shields.io/docker/cloud/automated/deepaiorg/tf-adain-style-transfer.svg?style=plastic' />
        <img src='https://img.shields.io/docker/cloud/build/deepaiorg/tf-adain-style-transfer.svg' />
    </a>
</p>

This model has been integrated with [ai_integration](https://github.com/deepai-org/ai_integration/blob/master/README.md) for seamless portability across hosting providers.

# Overview

Implementation of Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization in Tensorflow

<p align='center'>
    <img src='input/content/cornell.jpg' height="200px">
    <img src='input/style/woman_with_hat_matisse.jpg' height="200px">
    <img src='output/cornell_stylized_woman_with_hat_matisse.jpg' height="200px">
</p>

# For details see [Fast Style Transfer](https://deepai.org/machine-learning-model/fast-style-transfer ) on [Deep AI](https://deepai.org)

## Paper
[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://deepai.org/publication/arbitrary-style-transfer-in-real-time-with-adaptive-instance-normalization)

Nvidia-Docker is required to run this image.

# Quick Start

docker pull deepaiorg/tf-adain-style-transfer

### HTTP
```bash
nvidia-docker run --rm -it -e MODE=http -p 5000:5000 deepaiorg/tf-adain-style-transfer
```
Open your browser to localhost:5000 (or the correct IP address)

### Command Line

Save your two images as content.jpg and style.jpg in the current directory.
```bash
nvidia-docker run --rm -it -v `pwd`:/shared -e MODE=command_line deepaiorg/tf-adain-style-transfer --content /shared/content.jpg --style /shared/style.jpg --output /shared/output.jpg
```
# Docker build
```bash
docker build -t tf-adain-style-transfer .
```
## Acknowledgement

- This implementation is based on the original Torch implementation (https://github.com/xunhuang1995/AdaIN-style)
