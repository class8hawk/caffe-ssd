#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''
from __future__ import print_function
import os
import sys
import easydict, time
import numpy as np
from PIL import Image, ImageDraw
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=5):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

def main(args):
    '''main '''
    print('准备加载测试图片列表')
    images = []
    results = []
    with open(args.test_list_path, 'r') as f:
        for name in f.readlines():
            images.append(os.path.join(args.test_image_path, name[:-1] + args.test_image_ext))
    print('测试图片列表解析完毕...\n')

    print('加载检测模型')
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    print('模型载入完毕, 开始检测...')

    cpu_start = time.clock()
    time_start = time.time()
    for image in images:
        result = detection.detect(image, 0.5, 6)
        results.append(result)
    cpu_end = time.clock()
    time_end = time.time()

    num = len(images)
    print('检测完毕，共：%4d 张' % num)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('cpu :', cpu_end - cpu_start)
    print('time:', time_end - time_start)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')

    os.system('rm -rf ' + args.test_image_label_path)
    os.system('mkdir -p ' + args.test_image_label_path)

    for ind in range(num):
        image = images[ind]
        # print(image)
        img = Image.open(image)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        # print('image width=%03d height=%03d' % (width, height))
        # item 数组长度是 7： 0-3 4 5-6
        for item in results[ind]:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
            display_text = '%s: %.4f' %(item[-1], item[-2])
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 255, 255)) # 白
            # draw.text([xmin+2, ymin+2], item[-1] + str(item[-2]), (255, 0, 0)) # 红
            draw.text([xmin + 2, ymin + 2], display_text, (255, 0, 0))  # 红
            # print('xmin=%3d, ymin=%3d, xmax=%3d,ymax=%3d, score=%.4f, label=%s' % (xmin, ymin, xmax, ymax, item[-2],item[-1]))
        img.save(os.path.join(args.test_image_label_path, os.path.split(image)[1]))
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~')


def parse_args():
    '''parse args'''
    args = easydict.EasyDict()
    args.gpu_id = 1
    args.labelmap_file = 'data/tky/labelmap_voc.prototxt'
    args.model_def = 'models/VGGNet/tky/SSD_300x300/deploy.prototxt'
    args.image_resize = 300
    args.model_weights = 'models/VGGNet/tky/SSD_300x300/VGG_tky_SSD_300x300_iter_12000.caffemodel'
    args.image_file = 'examples/images/fish-bike.jpg'
    # for test
    args.test_list_path = '/home/wanghao/data/TKY/VOC2007/ImageSets/Main/test.txt'
    args.test_image_path = '/home/wanghao/data/TKY/VOC2007/JPEGImages'
    args.test_image_ext = '.JPG'
    args.test_image_label_path = 'data/draw' # 保存结果位置
    return args

if __name__ == '__main__':
    main(parse_args())
