# -*- coding: utf-8 -*-
"""
说明： 在完成了在Linux 平台下 caffe环境的 编译配置之后, 
      也在终端中通过执行脚本完成了最简单示例demo的运行： mnist cifar10
      
      现在假定你已经在linux 终端通过脚本 跑过了mnist实验, 知道 原始mnist数据集下载脚本 + lmdb数据格式转换脚本 做了什么
      在此基础上, 通过编写 Python 脚本, 一步一步了解如何 定义&生成 自己的caffe网络 以及 深入训练的每个过程！
      
首先从 最简单的 fc全连接网络开始, 实现对 mnist 手写字符识别任务（10分类）
脚本执行路径： caffe环境 根路径
10000迭代结果
Iteration 10000, loss = 0.0945602
Iteration 10000, Testing net (#0)
    Test net output #0: acc = 0.9572
    Test net output #1: loss = 0.147647 (* 1 = 0.147647 loss)
"""
# 1. 将pycaffe加入到系统路径中
from __future__ import print_function
import sys
import os
root = './'
caffe_python_pkg = os.path.join(root, 'python')
print(caffe_python_pkg)
sys.path.insert(0, caffe_python_pkg)

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

# 定义一个最简单的全链接网络
def fcnet(lmdb, batch_size, include_acc=False):
    # 网络规范
    n = caffe.NetSpec()
    # 数据层
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 256), ntop=2, name='mnist')

    # ip： 内积,即 全连接层fc
    n.ip1 = L.InnerProduct(n.data, num_output=512, weight_filler=dict(type='xavier'))
    # ReLU 激活函数
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    # fc 全连接层
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    # softmax层： 损失函数
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)

    # Accuracy 层
    if include_acc:
        n.acc = L.Accuracy(n.ip2, n.label)
        return n.to_proto()
    return n.to_proto()

# 创建&写入 *.prototxt
def write_net(train_proto, test_proto):
    with open(train_proto, 'w') as f:
        f.write(str(fcnet('examples/mnist/mnist_train_lmdb', 64)))

    with open(test_proto, 'w') as f:
        f.write(str(fcnet('examples/mnist/mnist_test_lmdb', 100, True)))

# 创建 solver.
def write_solver(solver_proto, train_proto, test_proto):

    solver = caffe_pb2.SolverParameter()
    # 给出 训练网络
    solver.train_net = train_proto
    # 给出 测试网络
    solver.test_net.append(test_proto)

    # 给出 测试迭代册数 10000/100
    solver.test_iter.append(100)
    # 测试间隔： 训练过程中每迭代N次, 执行一次测试
    solver.test_interval = 500

    solver.base_lr = 0.03
    solver.weight_decay = 5e-4
    solver.lr_policy = 'inv'
    solver.gamma = 1e-4
    solver.power = 0.75

    # 每迭代 100 次 显示一次结果
    solver.display = 100
    # 共计迭代训练 10000 次
    solver.max_iter = 10000
    # 每 5000 次 保存一次模型训练过程 快照(snapshot)
    solver.snapshot = 5000
    # 快照前缀： [存储路径]+[快照名字前缀]
    solver.snapshot_prefix = "examples/mnist/fcnet"
    # 训练模式： 这里是 GPU 模式
    solver.solver_mode = caffe_pb2.SolverParameter.GPU

    with open(solver_proto, 'w') as f:
        # f.write(str(solver))
        print(solver, file=f)

def train(solver_proto):
    # 0 号GPU模式，开始训练
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_proto)
    solver.solve()


if __name__ == '__main__':
    train_proto = 'examples/mnist/fcnet_train.prototxt'
    test_proto = 'examples/mnist/fcnet_test.prototxt'
    solver_proto = 'examples/mnist/fcnet_solver.prototxt'

    # 生成训练测试网络&solver训练超参数 PROTOTXT
    write_net(train_proto, test_proto)
    write_solver(solver_proto, train_proto, test_proto)

    # 开始训练
    train(solver_proto)