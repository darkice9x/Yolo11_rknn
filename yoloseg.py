'''
어딘가에서 로그 레벨 "WARNING" 문자열이 그대로 logger.setLevel()로 전달되고 있습니다.

torch 내부 코드 중 일부는 다음을 호출합니다.

logger.setLevel(level)


여기서 level == "WARNING" 이면,
Python 3.11 / 3.12 모두에서 아래 예외가 발생할 수 있습니다.

ValueError: Unknown level: 'WARNING'


torch 자체가 레벨 문자열을 만든 것이 아니라
사용자 환경 / 코드 / 환경변수에서 이미 오염됨
'''
#####################################################################
import logging
import os

# --- torch import 보호 ---
level = os.environ.get("LOGLEVEL", None)
if isinstance(level, str):
    level = logging._nameToLevel.get(level.upper(), logging.WARNING)

logging.basicConfig(
    level=level or logging.WARNING or logging.DEBUG,
    format='%(name)s : %(message)s',
)
import torch
#####################################################################
import cv2
import numpy as np
from numpy import dot, sqrt
import time
import numpy as np
import torchvision
import torch.nn.functional as F
from utils import *
from rknnlite.api import RKNNLite

COCO = 0
FIRE = 1
CRACK = 2
LICENSE = 3
GARBAGE = 4

def hex2rgb(h):  # rgb order (PIL)
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

class YoloSeg(object):
    def __init__(self,
                 #yolo_model: str,
                 RK3588_RKNN_MODEL: str,
                 input_size = 640,
                 DATASET = COCO,
                 NMS_THRESH = 0.45,
                 OBJ_THRESH = 0.25,
                 MAX_DETECT = 300
                 ) -> None:
        #self.yolo_model = yolo_model
        self.rknn_lite = RKNNLite()
        self.input_size = input_size
        self.NMS_THRESH = NMS_THRESH
        self.OBJ_THRESH = OBJ_THRESH
        self.MAX_DETECT = MAX_DETECT
        self.DATASET = DATASET
        #self.logger = logging.getLogger("YOLO")
        #self.logger.setLevel(logging.DEBUG)
        #logging.basicConfig(format='%(name)s : %(message)s', level=logging.DEBUG)

        if self.DATASET == COCO :
            self.CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
                    "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
                    "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
                    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
                    "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
                    "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")
        elif self.DATASET == CRACK :
            self.CLASSES = ("crack", "")
        elif self.DATASET == GARBAGE :
            self.CLASSES = ("Aluminium foil", "Battery", "Aluminium blister pack", "Carded blister pack", "Other plastic bottle",
                    "Plastic bottle", "Clear plastic bottle", "Glass bottle", "Plastic bottle cap", "Metal bottle cap", "Broken glass",
                    "Food Can", "Aerosol", "Drink can", "Toilet tube", "Other carton", "Egg carton", "Drink carton", "Corrugated carton",
                    "Meal carton", "Pizza box", "Paper cup", "Disposable plastic cup", "Foam cup", "Glass cup", "Other plastic cup",
                    "Food waste", "Glass jar", "Plastic lid", "Metal lid", "Other plastic", "Magazine paper", "Tissues", "Wrapping paper",
                    "Normal paper", "Paper bag", "Plastified paper bag", "Plastic film", "Six pack rings", "Garbage bag", "Other plastic wrapper",
                    "Single-use carrier bag", "Polypropylene bag", "Crisp packet", "Spread tub", "Tupperware", "Disposable food container",
                    "Foam food container", "Other plastic container", "Plastic glooves", "Plastic utensils", "Pop tab", "Rope & strings",
                    "Scrap metal", "Shoe", "Squeezable tube", "Plastic straw", "Paper straw", "Styrofoam piece", "Unlabeled litter", "Cigarette")

        #print('--> Load YOLO model')
        print_info(f'--> Load YOLO model')
        ret = self.rknn_lite.load_rknn(RK3588_RKNN_MODEL)
        if ret != 0:
            #print('Load RKNN model failed')
            print_info(f'Load RKNN model failed')
            exit(ret)
        print('done')

        #print('--> Init runtime environment YOLO')
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
            img_org = cv2.imread(input)
        else:
            img_org = input

        return self.seg(img_org)

    def letterbox_reverse_box(self, x1, y1, ratio):
        y1 = int(y1/ratio)
        x1 = int(x1/ratio)

        return [x1, y1]
    
    def letterbox(self, image, bg_color):
        """
        letterbox the image according to the specified size
        :param image: input image, which can be a NumPy array or file path
        :param size: target size (width, height)
        :param bg_color: background filling data 
        :return: processed image
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        target_width, target_height = self.input_size, self.input_size
        image_height, image_width, _ = image.shape

        # Calculate the adjusted image size
        aspect_ratio = min(target_width / image_width, target_height / image_height)
        self.new_width = int(image_width * aspect_ratio)
        self.new_height = int(image_height * aspect_ratio)

        # Use cv2.resize() for proportional scaling
        image = cv2.resize(image, (self.new_width, self.new_height), interpolation=cv2.INTER_AREA)

        # Create a new canvas and fill it
        result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
        #offset_x = (target_width - new_width) // 2
        #offset_y = (target_height - new_height) // 2
        result_image[0:self.new_height, 0:self.new_width] = image
        return result_image, aspect_ratio
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def filter_boxes(self, boxes, box_confidences, box_class_probs, seg_part):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score * box_confidences >= self.OBJ_THRESH)
        scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

        return boxes, classes, scores, seg_part

    def softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        n,c,h,w = position.shape
        p_num = 4
        mc = c//p_num
        y = position.reshape(n,p_num,mc,h,w)
        y = self.softmax(y, 2)
        acc_metrix = np.array(range(mc), dtype=float).reshape(1,1,mc,1,1)
        y = (y*acc_metrix).sum(2)
        return y
        #return y.numpy()

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.input_size//grid_h, self.input_size //grid_w]).reshape(1, 2, 1, 1)

        position = self.dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    def post_process(self, input_data):
        # input_data[0], input_data[4], and input_data[8] are detection box information
        # input_data[1], input_data[5], and input_data[9] are category score information
        # input_data[2], input_data[6], and input_data[10] are confidence score information
        # input_data[3], input_data[7], and input_data[11] are segmentation information
        # input_data[12] is the proto information
        proto = input_data[-1]
        boxes, scores, classes_conf, seg_part = [], [], [], []
        defualt_branch=3
        pair_per_branch = len(input_data)//defualt_branch

        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))
            seg_part.append(input_data[pair_per_branch*i+3])

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]
        seg_part = [sp_flatten(_v) for _v in seg_part]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)
        seg_part = np.concatenate(seg_part)

        # filter according to threshold
        boxes, classes, scores, seg_part = self.filter_boxes(boxes, scores, classes_conf, seg_part)

        zipped = zip(boxes, classes, scores, seg_part)
        sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
        result = zip(*sort_zipped)

        max_nms = 30000
        n = boxes.shape[0]  # number of boxes
        if not n:
            return None, None, None, None
        elif n > max_nms:  # excess boxes
            boxes, classes, scores, seg_part = [np.array(x[:max_nms]) for x in result]
        else:
            boxes, classes, scores, seg_part = [np.array(x) for x in result]

        # nms
        nboxes, nclasses, nscores, nseg_part = [], [], [], []
        agnostic = 0
        max_wh = 7680
        c = classes * (0 if agnostic else max_wh)
        ids = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float32) + torch.tensor(c, dtype=torch.float32).unsqueeze(-1),
                                torch.tensor(scores, dtype=torch.float32), self.NMS_THRESH)
        real_keeps = ids.tolist()[:self.MAX_DETECT]
        nboxes.append(boxes[real_keeps])
        nclasses.append(classes[real_keeps])
        nscores.append(scores[real_keeps])
        nseg_part.append(seg_part[real_keeps])

        if not nclasses and not nscores:
            return None, None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        seg_part = np.concatenate(nseg_part)

        ph, pw = proto.shape[-2:]
        proto = proto.reshape(seg_part.shape[-1], -1)
        seg_img = np.matmul(seg_part, proto)
        seg_img = self.sigmoid(seg_img)
        seg_img = seg_img.reshape(-1, ph, pw)

        seg_threadhold = 0.5

        # crop seg outside box
        seg_img = F.interpolate(torch.tensor(seg_img)[None], torch.Size([640, 640]), mode='bilinear', align_corners=False)[0]
        seg_img_t = self._crop_mask(seg_img,torch.tensor(boxes) )

        seg_img = seg_img_t.numpy()
        seg_img = seg_img > seg_threadhold
        return boxes, classes, scores, seg_img

    def draw(self, image, boxes, scores, classes, label=False):
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            print("%s @ (%d %d %d %d) %.3f" % (self.CLASSES[cl], top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            if label:
                cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
                            (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _crop_mask(self, masks, boxes):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """

        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
        
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def merge_seg(self, image, seg_img_org, classes):
        color = Colors()
        #print(f'seg_img_org shape: {seg_img_org.shape}')
        seg_img = seg_img_org[:, 0:self.new_height, 0:self.new_width]
        #print(f'seg_img shape: {seg_img.shape}')
        seg_img = np.where(seg_img, 1, 0).astype(np.uint8).transpose(1,2,0)
        h, w = image.shape[:2]
        seg_img = cv2.resize(seg_img, (w , h), interpolation=cv2.INTER_LINEAR)
        #print(f'seg_img shape: {seg_img.shape}')
        #print(f'seg_img shape: {(int(self.new_height/self.ratio) , int(self.new_width/self.ratio))}')
        if len(seg_img.shape) < 3:
            seg_img = seg_img[None,:,:]
        else:
            seg_img = seg_img.transpose(2,0,1)

        for i in range(len(seg_img)):
            seg = seg_img[i]
            seg = seg.astype(np.uint8)
            seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
            seg = seg * color(classes[i])
            seg = seg.astype(np.uint8)
            try:
                #image = cv2.add(image, seg)
                ''''
                alpha = 0.6  # seg 투명도 (0.0 ~ 1.0)
                beta = 1.0    # image 투명도 (보통 1.0)

                result = cv2.addWeighted(image, beta, seg, alpha, 0)
                '''
                seg_blurred = cv2.GaussianBlur(seg, (15, 15), 0)
                image = cv2.addWeighted(image, 1.0, seg_blurred, 0.4, 0)
            except cv2.error as e:
                continue

        return image
    
    def seg(self, img_org):
        img, self.ratio = self.letterbox(img_org, 114)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[img])
        #print( f'inference time : {(time.time()-start_time)*1000}ms')
        self.infertime = (time.time() - start_time)*1000
        print_info(f'inference time : {self.infertime}ms')
        boxes, classes, scores, seg_img = self.post_process(outputs)
        return boxes, classes, scores, seg_img
    
    def draw(self, image, boxes, scores, classes, classname, label=False):
        """Draw the boxes on the image.

        # Argument:
            image: original image.
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
            all_classes: all classes name.
        """
        for box, score, cl in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            #print('class: {}, score: {}'.format(CLASSES[cl], score))
            #print('letter box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
            #print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top-dh, left-dw, right-dw, bottom-dw))
            x1, y1 = self.letterbox_reverse_box(x1, y1, self.ratio)
            x2, y2 = self.letterbox_reverse_box(x2, y2, self.ratio)

            if classname == "all":
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if label:
                    cv2.putText(image, '#{0} {1:.2f}'.format(self.CLASSES[cl], score),
                                (x1 , y2 - 6),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                0.9, (0, 0, 255), 2)

    def info(self):
        print_info(f'Inference time: {self.infertime} ms')
        
    def release(self):
        self.rknn_lite.release()
        print_info(f'RKNN Seg Release!!')
