#!/bin/sh
docker run -it --rm --gpus all -p 8888:8888 --name anonym $(cat IMAGE)
