#!/bin/bash

IMAGE_NAME='gen-description'
docker run --gpus all -p 7860:7860 $IMAGE_NAME