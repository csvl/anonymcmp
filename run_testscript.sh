#!/bin/sh
docker run --rm --gpus all -v ${PWD}/tests/results:/tf/dev/tests/results $(cat IMAGE) \
 /bin/sh -c 'cd /tf/dev/tests; python anonymization_adult-NN1.py'
