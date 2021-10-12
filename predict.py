import os
import torch
import cv2
from PIL import Image

from model import LineNet, get_model
from dataset import preprocess, preprocess1
from torchvision import transforms
import torch.nn.functional as F

import shutil
import numpy as np


class Recognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.labels = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]

        src_size = (1376, 1104)  # w, h
        dst_h = 512
        dst_w = int(src_size[0] / src_size[1] * dst_h)
        dst_w = dst_w - dst_w % 4  # 四的倍数
        self.dstSize = (dst_w, dst_h)

        self.dstSize = (512, 512)
        # pretrained_model_path = 'saved_models/e11_val-acc_0.5372.pt'
        # pretrained_model_path = 'saved_models/e26_val-acc_0.6605.pt'
        # pretrained_model_path = 'baseline/上传的权重/model_best.pt'
        pretrained_model_path = 'saved_models/1009/e88_lr_0.000039_train-acc_0.8612_val-acc_0.0000.pt'

        # self.model = LineNet(len(self.labels), self.dstSize)
        self.model = get_model(len(self.labels), False)

        self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transforms = transforms.Compose([transforms.CenterCrop(896)])
        self.transforms_five = transforms.Compose([transforms.FiveCrop(896)])
        self.transforms_ten = transforms.Compose([transforms.TenCrop(896)])

    def predict(self, img):
        # img_trans = preprocess(img, self.dstSize, self.transforms)
        # img_trans = torch.Tensor([img_trans])
        img_trans = preprocess1(img, self.dstSize, self.transforms_ten)
        img_trans = torch.Tensor(img_trans)

        img_trans = img_trans.to(device=self.device, dtype=torch.float32)

        outputs = self.model(img_trans)

        # res_softmax = F.softmax(outputs[0], dim=0).to('cpu')
        # conf, pred = res_softmax.topk(1, 0, True, True)
        # confidence = conf.item()
        # label = self.labels[pred.item()]

        res_softmax = F.softmax(outputs, dim=1).to('cpu')
        res_softmax = res_softmax.detach().cpu().numpy()

        res_np = np.mean(res_softmax, axis=0)
        index = np.argmax(res_np)
        confidence = res_np[index]
        label = self.labels[index]

        return confidence, label


if __name__ == '__main__':
    # img_path = '/media/djb/E/hwjs/data/val'
    img_path = '/media/djb/E/hwjs/data/src'
    # img_path = '/media/djb/E/hwjs/data/src_sorted'
    categories = os.listdir(img_path)

    recognizer = Recognizer()

    right_count = 0  # 完全正确的数量
    right_count1 = 0  # 偏差±0.5级
    total_count = 0
    max_error = 0
    for category in categories:
        category_path = os.path.join(img_path, category)
        category_float = float(category)
        files = os.listdir(category_path)

        if not os.path.exists(os.path.join('/media/djb/E/hwjs/data/fliter/', category)):
            os.mkdir(os.path.join('/media/djb/E/hwjs/data/fliter/', category))

        for file in files:
            # img = cv2.imread(os.path.join(category_path, file))
            img = Image.open(os.path.join(category_path, file))
            ret, pred = recognizer.predict(img)

            error = abs(pred - category_float)
            if error < 0.01:
                right_count += 1
                right_count1 += 1
            elif error <= 0.51:
                right_count1 += 1
                print('label ', category_float, ' pred ', pred, 'error = ', error)
            else:
                if error > max_error:
                    max_error = error
                # shutil.copyfile(os.path.join(category_path, file), '/media/djb/E/hwjs/data/fliter/' + category + '/' + file)
                print('label ', category_float, ' pred ', pred, 'error = ', error)

            total_count += 1
            if total_count % 10 == 0:
                print('(right {}, right1 {}, count {},  accuracy = {}, max error {}'.format(right_count, right_count1, total_count, (0.4 * right_count + 0.6 * right_count1) / total_count, max_error))

        print('(right {}, right1 {}, count {},  accuracy = {}, max error {}'.format(right_count, right_count1,
                                                                                    total_count, (0.4 * right_count + 0.6 * right_count1) / total_count,
                                                                                    max_error))

