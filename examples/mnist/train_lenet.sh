#!/bin/bash
set -x
set -e
LOG="examples/mnist/log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
echo Logging output to "$LOG"

exec 2> >(tee  -a $LOG )

./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
