import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import cv2,math
import copy
from math import ceil

from itertools import product as product
from shapely.geometry import Polygon
from utils import *
#from rknn.api import RKNN
from rknnlite.api import RKNNLite

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax,angle):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.angle=angle

class YoloOBB(object):
    def __init__(self,
                 RKNN_MODEL: str,
                 input_size=640,
                 NMS_THRESH=0.4,
                 OBJ_THRESH=0.5,
                 #plot_angle = True
                 ) -> None:
        
        self.OBJ_THRESH = OBJ_THRESH
        self.NMS_THRESH = NMS_THRESH
        self.rknn_lite = RKNNLite()
        self.input_size = input_size
        self.img_org = None
        self.CLASSES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 
                    'basketball court', 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter',
                    'roundabout', 'soccer ball field', 'swimming pool']
        #self.logger = logging.getLogger("YOLO")
        #logging.basicConfig(format='%(name)s : %(message)s', level=logging.DEBUG)

        # load RKNN model
        #print('--> Load RKNN model')
        print_info(f'--> Load YOLO model')
        ret = self.rknn_lite.load_rknn(RKNN_MODEL)
        if ret != 0:
            #print('Load RKNN model failed')
            print_info(f'Load RKNN model failed')
            exit(ret)
        print('done')

        # init runtime environment
        #print('--> Init runtime environment')
        print_info(f'--> Init runtime environment YOLO')
        # run on RK356x/RK3588 with Debian OS, do not need specify target.

        ret = self.rknn_lite.init_runtime()

        if ret != 0:
            #print('Init runtime environment failed')
            print_info(f'Init runtime environment failed')
            exit(ret)
        print('done')

    def __call__(self, input, box_vis=False, angle=False):
        """
        Call the detect method to perform inference on the input image.
        :param input: Input image, which can be a NumPy array or file path.
        :return: Processed image with detected keypoints and bounding boxes.
        """
        if isinstance(input, str):
            self.img_org = cv2.imread(input)
        else:
            self.img_org = input

        predbox = self.obb( self.img_org )

        return predbox
    
    def letterbox(self, image, bg_color):
        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("Input image is None")

        image_height, image_width = image.shape[:2]

        aspect_ratio = min(self.input_size / image_width,
                        self.input_size / image_height)

        self.new_width = int(image_width * aspect_ratio)
        self.new_height = int(image_height * aspect_ratio)

        image = cv2.resize(image, (self.new_width, self.new_height),
                        interpolation=cv2.INTER_AREA)

        result_image = np.ones(
            (self.input_size, self.input_size, 3),
            dtype=np.uint8
        ) * bg_color

        result_image[0:self.new_height, 0:self.new_width] = image
        return result_image, aspect_ratio
    
    def _rotate_rectangle(self, x1, y1, x2, y2, a):
        # 计算中心点坐标
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # 将角度转换为弧度
        # a = math.radians(a)
        # 对每个顶点进行旋转变换
        x1_new = int((x1 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
        y1_new = int((x1 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)

        x2_new = int((x2 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
        y2_new = int((x2 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

        x3_new = int((x1 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
        y3_new = int((x1 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

        x4_new =int( (x2 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
        y4_new =int( (x2 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)
        return [(x1_new, y1_new), (x3_new, y3_new),(x2_new, y2_new) ,(x4_new, y4_new)]

    def _intersection(self, g, p):
        g=np.asarray(g)
        p=np.asarray(p)
        g = Polygon(g[:8].reshape((4, 2)))
        p = Polygon(p[:8].reshape((4, 2)))
        if not g.is_valid or not p.is_valid:
            return 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter/union

    def _nms(self, detectResult, aspect_ratio):
        predBoxs = []

        sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
        for i in range(len(sort_detectboxs)):
            xmin1 = sort_detectboxs[i].xmin/aspect_ratio
            ymin1 = sort_detectboxs[i].ymin/aspect_ratio
            xmax1 = sort_detectboxs[i].xmax/aspect_ratio
            ymax1 = sort_detectboxs[i].ymax/aspect_ratio
            classId = sort_detectboxs[i].classId
            angle = sort_detectboxs[i].angle
            p1=self._rotate_rectangle(xmin1, ymin1, xmax1, ymax1, angle)
            p1=np.array(p1).reshape(-1)
            
            if sort_detectboxs[i].classId != -1:
                predBoxs.append(sort_detectboxs[i])
                for j in range(i + 1, len(sort_detectboxs), 1):
                    if classId == sort_detectboxs[j].classId:
                        xmin2 = sort_detectboxs[j].xmin/aspect_ratio
                        ymin2 = sort_detectboxs[j].ymin/aspect_ratio
                        xmax2 = sort_detectboxs[j].xmax/aspect_ratio
                        ymax2 = sort_detectboxs[j].ymax/aspect_ratio
                        angle2 = sort_detectboxs[j].angle
                        p2=self._rotate_rectangle(xmin2, ymin2, xmax2, ymax2, angle2)
                        p2=np.array(p2).reshape(-1)
                        iou=self._intersection(p1, p2)
                        if iou > self.NMS_THRESH:
                            sort_detectboxs[j].classId = -1
        return predBoxs

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def process(self, out, model_w, model_h, stride, angle_feature, index, scale_w=1, scale_h=1):
        class_num = len(self.CLASSES)

        # 안전하게 1차원으로 변환
        angle_feature = np.asarray(angle_feature).reshape(-1)

        xywh = out[:, :64, :]
        conf = self._sigmoid(out[:, 64:, :]).reshape(-1)

        valid_idx = np.where(conf > self.OBJ_THRESH)[0]
        if valid_idx.size == 0:
            return []

        boxes = []
        hwc = np.arange(model_h * model_w * class_num)
        w_idx = hwc % model_w
        h_idx = (hwc // model_w) % model_h
        c_idx = hwc // (model_w * model_h)

        data = np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1)

        for ik in valid_idx:
            w, h, c = int(w_idx[ik]), int(h_idx[ik]), int(c_idx[ik])

            # softmax 처리
            xywh_ = xywh[0, :, h * model_w + w].reshape(1, 4, 16, 1)
            xywh_ = self._softmax(xywh_, axis=2)
            xywh_ = (xywh_ * data).sum(axis=2, keepdims=True).reshape(-1)

            xywh_add = xywh_[:2] + xywh_[2:]
            xywh_sub = (xywh_[2:] - xywh_[:2]) * 0.5

            # 안전한 angle_feature 인덱싱 (휴리스틱 포함)
            pos = int(index + h * model_w + w)
            if pos < 0 or pos >= angle_feature.size:
                # 후보 휴리스틱들 (순서대로 시도)
                if angle_feature.size == model_w * model_h:
                    # angle_feature가 위치별(H*W) 정보만 있던 경우
                    pos = int(h * model_w + w)
                elif angle_feature.size == class_num:
                    # 클래스별 angle인 경우
                    pos = int(c)
                else:
                    # 그 외에는 안전하게 래핑(modulo)
                    pos = pos % angle_feature.size

            angle = (float(angle_feature[pos]) - 0.25) * math.pi
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            x = xywh_sub[0] * cos_a - xywh_sub[1] * sin_a
            y = xywh_sub[0] * sin_a + xywh_sub[1] * cos_a

            cx = (x + w + 0.5) * stride
            cy = (y + h + 0.5) * stride
            w_box = xywh_add[0] * stride
            h_box = xywh_add[1] * stride

            xmin = (cx - w_box * 0.5) * scale_w
            ymin = (cy - h_box * 0.5) * scale_h
            xmax = (cx + w_box * 0.5) * scale_w
            ymax = (cy + h_box * 0.5) * scale_h

            boxes.append(DetectBox(int(c), float(conf[ik]), xmin, ymin, xmax, ymax, angle))

        return boxes
    
    def post_process(self, results, aspect_ratio):
        stride_index_map = {
            80: (8, 0),
            40: (16, 20 * 4 * 20 * 4),
            20: (32, 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2),
        }

        outputs = []
        angle_feature = results[-1]

        for x in results[:-1]:
            width = x.shape[2]
            if width not in stride_index_map:
                continue
            stride, index = stride_index_map[width]

            # (79, h*w) 형태로 미리 reshape
            feature = x.reshape(1, 79, -1)
            outputs.extend(
                self.process(feature, x.shape[3], width, stride, angle_feature, index)
            )

        return self._nms(outputs, aspect_ratio)
    
    def obb(self, img_org):
        letterbox_img, self.ratio = self.letterbox(img_org, 114)  # letterbox缩放
        infer_img = letterbox_img[..., ::-1]  # BGR2RGB
        infer_img = np.expand_dims(infer_img, 0)

        # Inference
        print_info('--> Running model')
        start_time = time.time()
        results = self.rknn_lite.inference(inputs=[infer_img])
        self.infertime = (time.time() - start_time)*1000
        print_info(f'Inference time: {self.infertime}ms')

        predbox = self.post_process(results, self.ratio)

        return predbox
    
    def draw(self, predbox):
        for index in range(len(predbox)):
            xmin = int((predbox[index].xmin)/self.ratio)
            ymin = int((predbox[index].ymin)/self.ratio)
            xmax = int((predbox[index].xmax)/self.ratio)
            ymax = int((predbox[index].ymax/self.ratio))
            classId = predbox[index].classId
            score = predbox[index].score
            angle = predbox[index].angle
            points=self._rotate_rectangle(xmin,ymin,xmax,ymax,angle)
            cv2.polylines(self.img_org, [np.asarray(points, dtype=int)], True,  (0, 255, 0), 1)

            ptext = (xmin, ymin)
            title= self.CLASSES[classId] + "%.2f" % score
            cv2.putText(self.img_org, title, ptext, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

        return self.img_org
    
    def info(self):
        print_info(f'Inference time: {self.infertime} ms')
        
    def release(self):
        self.rknn_lite.release()
        print_info(f'RKNN OBB Release!!')
