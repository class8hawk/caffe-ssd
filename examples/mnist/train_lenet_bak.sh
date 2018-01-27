#!/usr/bin/env sh
set -x
set -e
LOG="examples/mnist/log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
echo Logging output to "$LOG"

set +x
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@ 2>&1 | tee $LOG
