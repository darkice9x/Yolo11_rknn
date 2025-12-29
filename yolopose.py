import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import cv2,math
import copy
from math import ceil
from utils import *
#from rknn.api import RKNN
from rknnlite.api import RKNNLite

CLASSES = ['person']

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)
kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

class YoloPose(object):
    def __init__(self,
                RKNN_MODEL: str,
                input_size=320,
                NMS_THRESH=0.4,
                OBJ_THRESH=0.5,
                calc_angle = True
                ) -> None:

        self.rknn_lite = RKNNLite()
        self.input_size = input_size
        #self.box_score = box_score
        #self.kpt_score = kpt_score
        self.OBJ_THRESH = OBJ_THRESH
        self.NMS_THRESH = NMS_THRESH
        self.calc_angle = calc_angle
        self.left_knee_angle = 0 
        self.right_knee_angle = 0
        self.left_elbow_angle = 0
        self.right_elbow_angle = 0
        self.left_knee_list = []
        self.left_knee_list_sum = []
        self.right_knee_list = []
        self.right_knee_list_sum = []
        self.left_elbow_list = []
        self.left_elbow_list_sum = []
        self.right_elbow_list = []
        self.right_elbow_list_sum = []
        self.sum_len = 0
        #self.logger = logging.getLogger("YOLO")
        #self.logger.setLevel(logging.DEBUG)
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
            img_org = cv2.imread(input)
        else:
            img_org = input

        return self.pose(img_org, box_vis=box_vis, angle=angle)
    
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
    
    def _iou(self, xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        innerWidth = xmax - xmin
        innerHeight = ymax - ymin

        innerWidth = innerWidth if innerWidth > 0 else 0
        innerHeight = innerHeight if innerHeight > 0 else 0

        innerArea = innerWidth * innerHeight

        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        total = area1 + area2 - innerArea

        return innerArea / total


    def _nms(self, detectResult):
        predBoxs = []

        sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

        for i in range(len(sort_detectboxs)):
            xmin1 = sort_detectboxs[i].xmin
            ymin1 = sort_detectboxs[i].ymin
            xmax1 = sort_detectboxs[i].xmax
            ymax1 = sort_detectboxs[i].ymax
            classId = sort_detectboxs[i].classId

            if sort_detectboxs[i].classId != -1:
                predBoxs.append(sort_detectboxs[i])
                for j in range(i + 1, len(sort_detectboxs), 1):
                    if classId == sort_detectboxs[j].classId:
                        xmin2 = sort_detectboxs[j].xmin
                        ymin2 = sort_detectboxs[j].ymin
                        xmax2 = sort_detectboxs[j].xmax
                        ymax2 = sort_detectboxs[j].ymax
                        iou = self._iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                        if iou > self.NMS_THRESH:
                            sort_detectboxs[j].classId = -1
        return predBoxs


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x, axis=-1):
        # 将输入向量减去最大值以提高数值稳定性
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _process(self, out,keypoints,index,model_w,model_h,stride,scale_w=1,scale_h=1):
        xywh=out[:,:64,:]
        conf=self._sigmoid(out[:,64:,:])
        out=[]
        for h in range(model_h):
            for w in range(model_w):
                for c in range(len(CLASSES)):
                    if conf[0,c,(h*model_w)+w]>self.OBJ_THRESH:
                        xywh_=xywh[0,:,(h*model_w)+w] #[1,64,1]
                        xywh_=xywh_.reshape(1,4,16,1)
                        data=np.array([i for i in range(16)]).reshape(1,1,16,1)
                        xywh_=self._softmax(xywh_,2)
                        xywh_ = np.multiply(data, xywh_)
                        xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                        xywh_temp=xywh_.copy()
                        xywh_temp[0]=(w+0.5)-xywh_[0]
                        xywh_temp[1]=(h+0.5)-xywh_[1]
                        xywh_temp[2]=(w+0.5)+xywh_[2]
                        xywh_temp[3]=(h+0.5)+xywh_[3]

                        xywh_[0]=((xywh_temp[0]+xywh_temp[2])/2)
                        xywh_[1]=((xywh_temp[1]+xywh_temp[3])/2)
                        xywh_[2]=(xywh_temp[2]-xywh_temp[0])
                        xywh_[3]=(xywh_temp[3]-xywh_temp[1])
                        xywh_=xywh_*stride

                        xmin=(xywh_[0] - xywh_[2] / 2) * scale_w
                        ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                        xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                        ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                        keypoint=keypoints[...,(h*model_w)+w+index] 
                        keypoint[...,0:2]=keypoint[...,0:2]//1
                        box = DetectBox(c,conf[0,c,(h*model_w)+w], xmin, ymin, xmax, ymax,keypoint)
                        out.append(box)

        return out

    def _get_stride_index_map(self, image_size: int) -> dict:
        """
        입력 이미지 크기에 따라 stride_index_map을 자동 계산합니다.
        예: image_size=512 -> {64: (8, 0), 32: (16, 4096), 16: (32, 5120)}

        Args:
            image_size (int): 정사각형 입력 이미지 크기 (예: 512, 640 등)

        Returns:
            dict: {output_size: (stride, index_offset)} 형태의 stride index map
        """
        stride_list = [8, 16, 32]
        stride_index_map = {}

        offset = 0
        for stride in stride_list:
            output_size = image_size // stride
            num_elements = output_size * output_size
            stride_index_map[output_size] = (stride, offset)
            offset += num_elements

        return stride_index_map
    
    def _postprocess(self, results, size):
        outputs = []
        keypoints=results[3]
        # stride와 index를 미리 정의해둘 수 있는 경우 딕셔너리로 처리
        stride_index_map = self._get_stride_index_map(size)

        for x in results[:3]:
            width = x.shape[2]
            if width not in stride_index_map:
                index, stride = 0, 0
                #continue
            else:
                stride, index = stride_index_map[width]

            feature = x.reshape(1, 65, -1)
            output = self._process(feature, keypoints, index, x.shape[3], width, stride)
            outputs.extend(output)

        predbox = self._nms(outputs)
        #print(f'predbox: {len(predbox)}')
        return predbox
    
    def pose( self, img_org, box_vis = False, angle=False ):
        letterbox_img, self.ratio = self.letterbox(img_org, 56)  # letterbox缩放
        infer_img = letterbox_img[..., ::-1]  # BGR2RGB
        infer_img = np.expand_dims(infer_img, axis=0)  # add batch dimension
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[infer_img])
        self.infertime = (time.time() - start_time)*1000
        print_info(f'Inference time: {self.infertime} ms')

        predbox = self._postprocess(outputs, size=self.input_size)

        if self.calc_angle :
            lpredbox = copy.deepcopy(predbox)

            for bbox in lpredbox :
                kpts =bbox.keypoint.reshape(-1, 3) #keypoint [x, y, conf]
                #kpts = np.round(kpts / self.ratio).astype(int)
                #kpts[...,0]=np.round(kpts[...,0]/self.ratio).astype(int)
                #kpts[...,1]=np.round(kpts[...,1]/self.ratio).astype(int)
                kpts[..., :2] = np.round(kpts[..., :2] / self.ratio).astype(np.int32)
                #print(f'keypoints: {kpts}')
                #np.set_printoptions(suppress=True)
                #print(f'kpts : {kpts}')
                # 12, 14, 16
                #self.left_knee_angle = self._calculateAngle((int( kpts[12][0]), int(kpts[12][1])), (int( kpts[14][0]), int(kpts[14][1])), (int( kpts[16][0]), int(kpts[16][1])))
                self.left_knee_angle = self._calculateAngle((kpts[12][0], kpts[12][1]), (kpts[14][0], kpts[14][1]), (kpts[16][0], kpts[16][1]))
                self.left_knee_list_sum.append(self.left_knee_angle)
                self.left_knee_list_sum = self.left_knee_list_sum[-10:]
                self.sum_len = len( self.left_knee_list_sum  )
                self.left_knee_angle = int(sum(self.left_knee_list_sum)/self.sum_len)
                self.left_knee_list.append(self.left_knee_angle)
                self.left_knee_list = self.left_knee_list[-50:]
                # 11, 13, 15
                #self.right_knee_angle = self._calculateAngle((int( kpts[11][0]), int(kpts[11][1])), (int( kpts[13][0]), int(kpts[13][1])), (int( kpts[15][0]), int(kpts[15][1])))
                self.right_knee_angle = self._calculateAngle((kpts[11][0], kpts[11][1]), (kpts[13][0], kpts[13][1]), (kpts[15][0], kpts[15][1]))
                self.right_knee_list_sum.append(self.right_knee_angle)
                self.right_knee_list_sum = self.right_knee_list_sum[-10:]
                self.sum_len = len( self.right_knee_list_sum  )
                self.right_knee_angle = int(sum(self.right_knee_list_sum)/self.sum_len)
                self.right_knee_list.append(self.right_knee_angle)
                self.right_knee_list = self.right_knee_list[-50:]

                # 6, 8, 10
                #self.left_elbow_angle = self._calculateAngle((int( kpts[6][0]), int(kpts[6][1])), (int( kpts[8][0]), int(kpts[8][1])), (int( kpts[10][0]), int(kpts[10][1])))
                self.left_elbow_angle = self._calculateAngle((kpts[6][0], kpts[6][1]), (kpts[8][0], kpts[8][1]), (kpts[10][0], kpts[10][1]))
                self.left_elbow_list_sum.append(self.left_elbow_angle)
                self.left_elbow_list_sum = self.left_elbow_list_sum[-10:]
                self.sum_len = len( self.left_elbow_list_sum  )
                self.left_elbow_angle = int(sum(self.left_elbow_list_sum)/self.sum_len)
                self.left_elbow_list.append(self.left_elbow_angle)
                self.left_elbow_list = self.left_elbow_list[-50:]
                # 5, 7, 9
                #self.right_elbow_angle = self._calculateAngle((int( kpts[5][0]), int(kpts[5][1])), (int( kpts[7][0]), int(kpts[7][1])), (int( kpts[9][0]), int(kpts[9][1])))
                self.right_elbow_angle = self._calculateAngle((kpts[5][0], kpts[5][1]), (kpts[7][0], kpts[7][1]), (kpts[9][0], kpts[9][1]))
                self.right_elbow_list_sum.append(self.right_elbow_angle)
                self.right_elbow_list_sum = self.right_elbow_list_sum[-10:]
                self.sum_len = len( self.right_elbow_list_sum  )
                self.right_elbow_angle = int(sum(self.right_elbow_list_sum)/self.sum_len)
                self.right_elbow_list.append(self.right_elbow_angle)
                self.right_elbow_list = self.right_elbow_list[-50:]

                #print(f'left_knee_angle: {self.left_knee_angle}')
                #print(f'right_knee_angle: {self.right_knee_angle}')
                #print(f'left_elbow_angle: {self.left_elbow_angle}')
                #print(f'right_elbow_angle: {self.right_elbow_angle}')

        #draw_img = self.draw(img_org, predbox, aspect_ratio, box_vis = box_vis, angle=angle)
        return predbox

    def draw(self, img, predbox, box_vis = False, angle = False):
        for i in range(len(predbox)):
            xmin = int((predbox[i].xmin)/self.ratio)
            ymin = int((predbox[i].ymin)/self.ratio)
            xmax = int((predbox[i].xmax)/self.ratio)
            ymax = int((predbox[i].ymax)/self.ratio)
            classId = predbox[i].classId
            score = predbox[i].score
            if box_vis :
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                ptext = (xmin, ymin)
                title= CLASSES[classId] + "%.2f" % score

                cv2.putText(img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            keypoints =predbox[i].keypoint.reshape(-1, 3) #keypoint [x, y, conf]
            keypoints[..., :2] = np.round(keypoints[..., :2] / self.ratio).astype(np.int32)

            for k, keypoint in enumerate(keypoints):
                x, y, conf = keypoint.astype(np.int32)
                x, y = int(round(x)), int(round(y))
                color_k = tuple(int(x) for x in kpt_color[k])
                if x != 0 and y != 0:
                    
                    cv2.circle(img, (x, y), 3, color_k, -1, lineType=cv2.LINE_AA)
                    if angle:
                        if k == 14:
                            cv2.putText(img, f'{self.left_knee_angle}', (x+10, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                        if k == 13:
                            cv2.putText(img, f'{self.right_knee_angle}', (x+10, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                        if k == 8:
                            cv2.putText(img, f'{self.left_elbow_angle}', (x+10, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                        if k == 7:
                            cv2.putText(img, f'{self.right_elbow_angle}', (x+10, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            for k, sk in enumerate(skeleton):
                pos1 = (
                    int(keypoints[sk[0] - 1, 0]),
                    int(keypoints[sk[0] - 1, 1])
                )
                pos2 = (
                    int(keypoints[sk[1] - 1, 0]),
                    int(keypoints[sk[1] - 1, 1])
                )

                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                cv2.line(img, pos1, pos2, tuple(int(x) for x in limb_color[k]), thickness=2, lineType=cv2.LINE_AA)

        return img
    
    def _calculateAngle(self, landmark1, landmark2, landmark3):
        '''
        This function calculates angle between three different landmarks.
        Args:
            landmark1: The first landmark containing the x,y and z coordinates.
            landmark2: The second landmark containing the x,y and z coordinates.
            landmark3: The third landmark containing the x,y and z coordinates.
        Returns:
            angle: The calculated angle between the three landmarks.

        '''

        # Get the required landmarks coordinates.
        x1, y1 = landmark1
        x2, y2 = landmark2
        x3, y3 = landmark3

        # Calculate the angle between the three points
        angle = abs( math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)))

        # Check if the angle is less than zero.
        if angle > 180.0:

            # Add 360 to the found angle.
            angle = 360 - angle
        
        # Return the calculated angle.
        return angle
    
    def info(self):
        print_info(f'Inference time: {self.infertime} ms')
        
    def release(self):
        self.rknn_lite.release()
        print_info(f'RKNN Pose Release!!')