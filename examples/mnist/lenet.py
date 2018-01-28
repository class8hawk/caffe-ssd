# -*- coding: utf-8 -*-
"""
说明： 在完成了在Linux 平台下 caffe环境的 编译配置之后, 
      也在终端中通过执行脚本完成了最简单示例demo的运行： mnist cifar10
      
      现在假定你已经在linux 终端通过脚本 跑过了mnist实验, 知道 原始mnist数据集下载脚本 + lmdb数据格式转换脚本 做了什么
      在此基础上, 通过编写 Python 脚本, 一步一步了解如何 定义&生成 自己的caffe网络 以及 深入训练的每个过程！
      
2.1 实现python下的 lenet 网络创建, 以及模型训练
train： 60,000  test： 10,000
脚本执行路径： caffe环境 根路径
10000迭代结果
Iteration 10000, loss = 0.00263483
Iteration 10000, Testing net (#0)
    Test net output #0: acc = 0.9914
    Test net output #1: loss = 0.0258369 (* 1 = 0.0258369 loss)
"""
# 1. 将pycaffe加入到系统路径中
from __future__ import print_function
import sys
import os
root = './'
caffe_python_pkg = os.path.join(root, 'python')
sys.path.insert(0, caffe_python_pkg)

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from pprint import pprint

# 定义一个最简单的全链接网络
def lenet(lmdb, batch_size, include_acc=False):
    kwargs = {
        'param': [dict(lr_mult=1), dict(lr_mult=2)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant')}
    # 网络规范
    net = caffe.NetSpec()
    # 数据层
    net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2, name='mnist')

    net.conv1 = L.Convolution(net.data, kernel_size=5, stride=1, num_output=20, **kwargs)
    net.pool1 = L.Pooling(net.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    net.conv2 = L.Convolution(net.pool1, kernel_size=5, stride=1, num_output=50, **kwargs)
    net.pool2 = L.Pooling(net.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # ip： 内积[InnerProduct],即 全连接层fc
    net.ip1 = L.InnerProduct(net.pool2, num_output=500, **kwargs)
    # ReLU 激活函数
    net.relu1 = L.ReLU(net.ip1, in_place=True)
    # fc 全连接层
    net.ip2 = L.InnerProduct(net.relu1, num_output=10, **kwargs)
    # softmax层： 损失函数
    net.loss = L.SoftmaxWithLoss(net.ip2, net.label)

    # Accuracy 层
    if include_acc:
        net.acc = L.Accuracy(net.ip2, net.label)
        return net.to_proto()
    return net.to_proto()

# 创建&写入 *.prototxt
def write_net(train_proto, test_proto):
    with open(train_proto, 'w') as f:
        f.write(str(lenet('examples/mnist/mnist_train_lmdb', 64)))

    with open(test_proto, 'w') as f:
        f.write(str(lenet('examples/mnist/mnist_test_lmdb', 100, True)))

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
    solver.momentum = 0.9
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
    solver.snapshot_prefix = "examples/mnist/lenet"
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
    # 自动完成迭代训练过程
    solver.solve()

def _train(solver_proto):
    # 0 号GPU模式，开始训练
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_proto)

    """
    solver.net.blobs: 数据&中间计算结果, 字典类型, key=layer_name
    solver.net.params: 网络模型参数 即: 模型weight + bias 字典类型, key=layer_name
    """
    print('________________________________________________________')
    print('| each output is (batch size, feature dim, spatial dim) |')
    print('========================================================')
    pprint([(k, v.data.shape) for k, v in solver.net.blobs.items()])

    print('________________________________________________________')
    print('| just print the weight sizes (we\'ll omit the biases)  |')
    print('========================================================')
    # weight & bias
    pprint([(k, v[0].data.shape, v[1].data.shape) for k, v in solver.net.params.items()])

    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    from pylab import *
    '''
    print(solver.net.forward())  # train net
    print(solver.test_nets[0].forward())  # test net (there can be more than one)
    
    # 显示blobs中的图像
    n=8 #-1
    imshow(solver.net.blobs['data'].data[:n, 0].transpose(1, 0, 2).reshape(28, -1), cmap='gray')
    axis('off')
    print('train labels:', solver.net.blobs['label'].data[:n])

    figure()
    imshow(solver.test_nets[0].blobs['data'].data[:n, 0].transpose(1, 0, 2).reshape(28, -1), cmap='gray');
    axis('off')
    print('test labels:', solver.test_nets[0].blobs['label'].data[:n])
    '''
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    solver.step(1)
    # figure()
    # model's weight to show; no bias
    # imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5).transpose(0, 2, 1, 3).reshape(4 * 5, 5 * 5), cmap='gray')
    # imshow(solver.net.params['conv1'][0].data[:, 0].reshape(4, 5, 5, 5).transpose(0, 2, 1, 3).reshape(4 * 5, 5 * 5), cmap='gray')
    # axis('off')
    feat = solver.net.blobs['conv1'].data[:1]
    imshow(feat.reshape(24, -1), cmap='gray')

    '''
    # the parameters are a list of [weights, biases]
    filters = solver.net.params['conv1'][0].data
    imshow(filters.reshape(5, -1),  cmap='gray')
    
    
    vis_square(filters)
    # caffe.vis_square(filters.transpose(0, 2, 3, 1))

    feat = solver.net.blobs['conv1'].data[0, :4]
    vis_square(feat)
    '''
    # a=solver.net.forward()  # train net
    # b=solver.test_nets[0].forward()  # test net (there can be more than one)
    # print(a)
    # print(b)

    # [先|只]迭代n=200次
    # solver.step(500)
    # 迭代完毕
    # solver.solve()

def __train(solver_proto):
    # 0 号GPU模式，开始训练
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_proto)

    niter = 200
    test_interval = 25
    # losses will also be stored in the log
    # import numpy as np
    from pylab import *
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter / test_interval)))
    output = zeros((niter, 8, 10))

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['ip2'].data[:8]

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            print('Iteration', it, 'testing...')
            correct = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
            test_acc[it // test_interval] = correct / 1e4

    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(niter), train_loss)
    ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))

    # for i in range(8):
    #     figure(figsize=(2, 2))
    #     imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    #     figure(figsize=(10, 2))
    #     imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    #     xlabel('iteration')
    #     ylabel('label')

if __name__ == '__main__':
    train_proto = 'examples/mnist/_lenet_train.prototxt'
    test_proto = 'examples/mnist/_lenet_test.prototxt'
    solver_proto = 'examples/mnist/_lenet_solver.prototxt'

    # 生成训练测试网络&solver训练超参数 PROTOTXT
    write_net(train_proto, test_proto)
    write_solver(solver_proto, train_proto, test_proto)

    # 开始训练
    _train(solver_proto)