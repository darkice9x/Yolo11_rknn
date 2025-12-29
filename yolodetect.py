import cv2
import numpy as np
from numpy import dot, sqrt
import time
from utils import *
from rknnlite.api import RKNNLite

COCO = 0
FIRE = 1
CRACK = 2
LICENSE = 3
GARBAGE = 4

class YoloDetect(object):
    def __init__(self,
                 #yolo_model: str,
                 RK3588_RKNN_MODEL: str,
                 input_size = 640,
                 class_agnostic = False,
                 DATASET = COCO,
                 NMS_THRESH = 0.45,
                 OBJ_THRESH = 0.25
                 ) -> None:
        #self.yolo_model = yolo_model
        self.rknn_lite = RKNNLite()
        self.input_size = input_size
        self.NMS_THRESH = NMS_THRESH
        self.OBJ_THRESH = OBJ_THRESH
        self.class_agnostic = class_agnostic
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
        elif self.DATASET == FIRE :
            self.CLASSES = ("fire", "", "smoke")
        elif self.DATASET == CRACK :
            self.CLASSES = ("crack", "")
        elif self.DATASET == LICENSE :
            self.CLASSES = ("license", "")

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

        boxes, classes, scores = self.detect(img_org)
        return boxes, classes, scores
    
    def letterbox_reverse_box(self, x1, y1, ratio):
        y1 = int(y1/ratio)
        x1 = int(x1/ratio)

        return [x1, y1]
    
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
    
    #def sigmoid(self, x):
    #    return 1 / (1 + np.exp(-x))

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape
        
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= self.OBJ_THRESH)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def _nms(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def _softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def _dfl(self, position):
        # Distribution Focal Loss (DFL)
        n,c,h,w = position.shape
        p_num = 4
        mc = c//p_num
        y = position.reshape(n,p_num,mc,h,w)
        y = self._softmax(y, 2)
        acc_metrix = np.array(range(mc), dtype=float).reshape(1,1,mc,1,1)
        y = (y*acc_metrix).sum(2)
        return y

    def _box_process(self, position, anchors):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.input_size//grid_h, self.input_size//grid_w]).reshape(1,2,1,1)

        position = self._dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    def _postprocess(self, input_data):
        values = [[[1.0,1.0]]]*3

        boxes, scores, classes_conf = [], [], []

        pair_per_branch = 3
        defualt_branch = len(input_data)//pair_per_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(self._box_process(input_data[pair_per_branch*i], None))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]
        
        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf)
    
        # nms
        nboxes, nclasses, nscores = [], [], []
        if self.class_agnostic:
            keep = self._nms(boxes, scores)
            if len(keep) != 0:
                nboxes.append(boxes[keep])
                nclasses.append(classes[keep])
                nscores.append(scores[keep])
        else:
            for c in set(classes):
                inds = np.where(classes == c)
                b = boxes[inds]
                c = classes[inds]
                s = scores[inds]
                keep = self._nms(b, s)

                if len(keep) != 0:
                    nboxes.append(b[keep])
                    nclasses.append(c[keep])
                    nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        boxes = (np.array(boxes)/self.ratio).astype(int)

        return boxes, classes, scores

    def draw(self, image, boxes, scores, classes, classname, outline=True, label=True):
        """Draw the boxes on the image.

        # Argument:
            image: original image.
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
            all_classes: all classes name.
        """
        count_index = 0
        rects = []
        for box, score, cl in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            #print('class: {}, score: {}'.format(CLASSES[cl], score))
            #print('letter box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
            #print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top-dh, left-dw, right-dw, bottom-dw))
            #x1, y1 = self.letterbox_reverse_box(x1, y1, self.ratio)
            #x2, y2 = self.letterbox_reverse_box(x2, y2, self.ratio)

            if classname == "all":
                if outline:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if label :
                    cv2.putText(image, '#{0} {1} {2:.2f}'.format(count_index, self.CLASSES[cl], score),
                                (x1 , y2 - 6),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                0.9, (0, 0, 255), 1)
                count_index += 1
                rects.append((int(x1), int(y1), int(x2), int(y2)))
            elif classname == "person":
                if self.CLASSES[cl] == "person":
                    if outline:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    if label :
                        cv2.putText(image, '#{0} {1} {2:.2f}'.format(count_index, self.CLASSES[cl], score),
                                    (x1, y2 - 6),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    0.9, (0, 0, 255), 1)
                    count_index += 1
                    rects.append((int(x1), int(y1), int(x2), int(y2)))
        
        return rects

    def detect(self, img_org):
        letterbox_img, self.ratio = self.letterbox(img_org, 114)  # letterbox缩放
        infer_img = letterbox_img[..., ::-1]  # BGR2RGB
        infer_img = np.expand_dims(infer_img, axis=0)  # add batch dimension
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[infer_img])
        self.infertime = (time.time() - start_time)*1000
        print_info(f'Inference time: {self.infertime} ms')

        boxes, classes, scores = self._postprocess(outputs)
        return boxes, classes, scores
    
    def info(self):
        print_info(f'Inference time: {self.infertime} ms')
        
    def release(self):
        self.rknn_lite.release()
        print_info(f'RKNN Detect Release!!')
                            
