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
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
root = './'
caffe_python_pkg = os.path.join(root, 'python')
sys.path.insert(0, caffe_python_pkg)

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from pprint import pprint

# 定义一个最简单的全链接网络
def alexnet(lmdb, mean_file, batch_size, include_acc=False):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'bias_filler': dict(type='constant', value=0)}
    # 网络规范
    net = caffe.NetSpec()
    # net.name = 'AlexNet'
    # 数据层
    net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(mirror=True, crop_size=227, mean_file=mean_file), ntop=2, name='ilsvrc12')

    net.conv1 = L.Convolution(net.data, kernel_size=11, stride=4, num_output=96, weight_filler=dict(type='gaussian', std=0.01), **kwargs)
    net.relu1 = L.ReLU(net.conv1, in_place=True)
    net.norm1 = L.LRN(net.relu1, lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75))
    net.pool1 = L.Pooling(net.norm1, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    net.conv2 = L.Convolution(net.pool1, kernel_size=5, pad=2, group=2, num_output=256, weight_filler=dict(type='gaussian', std=0.01), **kwargs)
    net.relu2 = L.ReLU(net.conv2, in_place=True)
    net.norm2 = L.LRN(net.relu2, lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75))
    net.pool2 = L.Pooling(net.norm2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    net.conv3 = L.Convolution(net.pool2, kernel_size=3, pad=1, num_output=384, weight_filler=dict(type='gaussian', std=0.01), **kwargs)
    net.relu3 = L.ReLU(net.conv3, in_place=True)

    net.conv4 = L.Convolution(net.relu3, kernel_size=3, pad=1, group=2, num_output=384, weight_filler=dict(type='gaussian', std=0.01), **kwargs)
    net.relu4 = L.ReLU(net.conv4, in_place=True)

    net.conv5 = L.Convolution(net.relu4, kernel_size=5, pad=2, group=2, num_output=256, weight_filler=dict(type='gaussian', std=0.01), **kwargs)
    net.relu5 = L.ReLU(net.conv5, in_place=True)
    net.pool5 = L.Pooling(net.relu5, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    #
    # 全连接层fc
    net.fc6 = L.InnerProduct(net.pool5, num_output=4096, weight_filler=dict(type='gaussian', std=0.005), **kwargs)
    net.relu6 = L.ReLU(net.fc6, in_place=True)
    net.drop6 = L.Dropout(net.relu6, dropout_param=dict(dropout_ratio=0.5))

    net.fc7 = L.InnerProduct(net.drop6, num_output=4096, weight_filler=dict(type='gaussian', std=0.005), **kwargs)
    net.relu7 = L.ReLU(net.fc7, in_place=True)
    net.drop7 = L.Dropout(net.relu7, dropout_param=dict(dropout_ratio=0.5))

    net.fc8 = L.InnerProduct(net.drop7, num_output=1000, weight_filler=dict(type='gaussian', std=0.01), **kwargs)

    # Accuracy 层
    if include_acc:
        net.accuracy = L.Accuracy(net.fc8, net.label)
        return net.to_proto()

    # softmax层： 损失函数
    net.loss = L.SoftmaxWithLoss(net.fc8, net.label)
    return net.to_proto()

# 创建&写入 *.prototxt
def write_net(train_proto, test_proto):
    mean_file = "examples/imagenet/imagenet_mean.binaryproto"
    with open(train_proto, 'w') as f:
        f.write(str(alexnet('examples/imagenet/ilsvrc12_train_lmdb', mean_file, 512))) # 256

    with open(test_proto, 'w') as f:
        f.write(str(alexnet('examples/imagenet/ilsvrc12_val_lmdb', mean_file, 500, True))) # 50

# 创建 solver.
def write_solver(solver_proto, train_proto, test_proto):

    solver = caffe_pb2.SolverParameter()
    # 给出 训练网络
    solver.train_net = train_proto
    # 给出 测试网络
    solver.test_net.append(test_proto)

    # 给出 测试迭代册数 50000/50
    solver.test_iter.append(100)
    # 测试间隔： 训练过程中每迭代N次, 执行一次测试
    solver.test_interval = 1000

    solver.base_lr = 0.01
    solver.lr_policy = 'step'
    solver.stepsize = 50000
    solver.gamma = 0.1
    solver.momentum = 0.9
    solver.weight_decay = 5e-4


    # 每迭代 20 次 显示一次结果
    solver.display = 20
    # 共计迭代训练 450000=45w 次
    solver.max_iter = 200000
    # 每 10000 次 保存一次模型训练过程 快照(snapshot)
    solver.snapshot = 10000
    # 快照前缀： [存储路径]+[快照名字前缀]
    solver.snapshot_prefix = "examples/imagenet/alexnet"
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
    # caffe.set_device(1)
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



    '''
    # the parameters are a list of [weights, biases]
    filters = solver.net.params['conv1'][0].data
    imshow(filters.reshape(5, -1),  cmap='gray')
    
    feat = solver.net.blobs['conv1'].data[:1]
    imshow(feat.reshape(24, -1), cmap='gray')
    
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

if __name__ == '__main__':
    train_proto = 'examples/imagenet/_alexnet_train.prototxt'
    test_proto = 'examples/imagenet/_alexnet_test.prototxt'
    solver_proto = 'examples/imagenet/_alexnet_solver.prototxt'

    # 生成训练测试网络&solver训练超参数 PROTOTXT
    # write_net(train_proto, test_proto)
    # write_solver(solver_proto, train_proto, test_proto)

    # 开始训练
    _train(solver_proto)