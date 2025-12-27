from yoloseg import *
from yolodetect import *
from yolopose import *    
from yoloobb import *


COCO = 1
FIRE = 2
CRACK = 3
LICENSE = 4
GARBAGE = 5

class Yolo(object):
    def __init__(self,
                 #yolo_model: str,
                 TASK: str,
                 RK3588_RKNN_MODEL: str,
                 input_size = 640,
                 DATASET = COCO,
                 DS_NMS_THRESH = 0.45,
                 DS_OBJ_THRESH = 0.25,
                 PO_NMS_THRESH = 0.4,
                 PO_OBJ_THRESH = 0.5,
                 MAX_DETECT = 300,
                 CALC_ANGLE = False
                 ) -> None:
        self.task = TASK
        self.detect = None
        self.pose = None
        self.seg = None
        self.obb = None
        
        if TASK == "detect" :
            self.detect = YoloDetect(RK3588_RKNN_MODEL, input_size = input_size, DATASET=DATASET)
        elif TASK == "pose" :
            self.pose = YoloPose(RK3588_RKNN_MODEL, input_size=input_size, calc_angle = CALC_ANGLE)
        elif TASK == "seg" :
            self.seg = YoloSeg(RK3588_RKNN_MODEL, input_size = input_size, DATASET=DATASET)
        elif TASK == "obb" :
            self.obb = YoloOBB(RK3588_RKNN_MODEL, input_size = input_size)

    def release(self):
        if self.detect is not None :
            self.detect.release()
        if self.pose is not None :
            self.pose.release()
        if self.seg is not None :
            self.seg.release()
        if self.obb is not None :
            self.obb.release()
