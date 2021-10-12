# -*- coding: utf-8 -*-
import os
import torch
import torchvision

from model_service.pytorch_model_service import PTServingBaseService

import time
from metric.metrics_manager import MetricsManager
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import log
from PIL import Image
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)

logger.info(torch.__version__)
logger.info(torchvision.__version__)


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path, **kwargs):
        self.model_name = model_name
        self.model_path = model_path
        # self.labels = self.read_classes()
        self.labels = ['6.5', '7.0', '7.5', '8.0', '8.5', '9.0', '9.5', '10.0', '10.5', '11.0', '11.5', '12.0', '12.5', '13.0']
        self.num_classes = len(self.labels)
        logger.info('{}-{}'.format(self.num_classes, self.model_path))
        for key in kwargs:
            logger.info('{}-{}'.format(key, kwargs[key]))

        self.model = self.initial_model(self.num_classes)

        self.use_cuda = False
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logger.info('Using GPU for inference')
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            logger.info('Using CPU for inference')
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()

        self.transforms = transforms.Compose([transforms.CenterCrop(896)])
        self.transforms_five = transforms.Compose([transforms.FiveCrop(896)])
        self.transforms_ten = transforms.Compose([transforms.TenCrop(896)])
        self.dst_size = (512, 512)

    def initial_model(self, num_classes):
        # model = torchvision.models.__dict__["resnet50"](pretrained=False)
        # channel_in = model.fc.in_features
        # model.fc = nn.Linear(channel_in, num_classes)
        # logger.info('{}-{}'.format(channel_in, num_classes))

        model = torchvision.models.resnet50(pretrained=False)
        fc_input = model.fc.in_features
        model.fc = nn.Linear(fc_input, num_classes)  # 修改最后一层全连接的输出类型

        return model

    # transform img before inference
    def transform_img(self, img):
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     normalize])

        img = self.transforms(img)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img_scaled = cv2.resize(img, self.dst_size, interpolation=cv2.INTER_AREA)

        # HWC to CHW
        img_trans = img_scaled.transpose((2, 0, 1))
        img_trans = img_trans / 255

        img_trans = torch.Tensor([img_trans])

        return img_trans

    def transform_img_five(self, img):
        # img = self.transforms_five(img)
        img = self.transforms_ten(img)

        results = []
        for img_sub in img:
            img_sub = cv2.cvtColor(np.asarray(img_sub), cv2.COLOR_RGB2BGR)
            img_scaled = cv2.resize(img_sub, self.dst_size, interpolation=cv2.INTER_AREA)

            # HWC to CHW
            img_trans = img_scaled.transpose((2, 0, 1))
            img_trans = img_trans / 255

            results.append(img_trans)

        results = torch.Tensor(results)

        return results

    def read_classes(self):
        labels = []
        class_path = os.path.join(os.path.dirname(self.model_path), 'classes.txt')
        with open(class_path) as lines:
            for line in lines:
                line = line.strip()
                labels.append(line)
        return labels

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content).convert("RGB")
                # img = self.transform_img(img)
                img = self.transform_img_five(img)
                # preprocessed_data[k] = torch.unsqueeze(img, dim=0).to(self.device)
                preprocessed_data[k] = img.to(self.device, dtype=torch.float32)
        return preprocessed_data

    def _inference(self, data):
        img = data["input_img"]
        data = img
        res = self.model(data)
        # res_softmax = F.softmax(res[0], dim=0).to('cpu')
        # conf, pred = res_softmax.topk(1, 0, True, True)
        # label = self.labels[pred.item()]

        res_softmax = F.softmax(res, dim=1).to('cpu')
        res_softmax = res_softmax.detach().cpu().numpy()
        res_np = np.mean(res_softmax, axis=0)
        index = np.argmax(res_np)
        confidence = res_np[index]
        label = self.labels[index]

        result = {}
        result['label'] = label
        # result['confidence'] = float('{0:.4f}'.format(conf.item()))
        result['confidence'] = float('{0:.4f}'.format(confidence))
        result = {"result": result}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data
