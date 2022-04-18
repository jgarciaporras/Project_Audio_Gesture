import os
import cv2
from base_camera import BaseCamera
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torchvision
import numpy as np
import argparse
# from utils.datasets import *
# from utils.utils import *

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import js2py

# import soundfile as sf
# import sounddevice as sd


# from pydub import AudioSegment
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# import math

import time


class Camera(BaseCamera):
    video_source = 'video/sample.mkv'

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        
        weights='weights/last.pt'  # model.pt path(s)
        source=0  # file/dir/URL/glob 0 for webcam
        data='data/coco128.yaml'  # dataset.yaml path
        imgsz=(416, 416)  # inference size (height width)
        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=1000  # maximum detections per image
        device=''  # cuda device i.e. 0 or 0123 or cpu
        view_img=False  # show results
        save_txt=False  # save results to *.txt
        save_conf=False  # save confidences in --save-txt labels
        save_crop=False  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        classes=None  # filter by class: --class 0 or --class 0 2 3
        agnostic_nms=False  # class-agnostic NMS
        augment=False  # augmented inference
        visualize=False  # visualize features
        update=False  # update all models
        project=ROOT / 'runs/detect'  # save results to project/name
        name='exp'  # save results to project/name
        exist_ok=False  # existing project/name ok do not increment
        line_thickness=3  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=False  # hide confidences
        half=False
        dnn=False

        # Get default audio device using PyCAW
        # devices = AudioUtilities.GetSpeakers()
        # interface = devices.Activate(
        # IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        # volume = cast(interface, POINTER(IAudioEndpointVolume))
        # # Get current volume 
        # currentVolumeDb = volume.GetMasterVolumeLevel()
        
        # filename = 'audio'
        # path = 'static/uploads/' + filename + '.wav' # path from my computer 
        # audio_file, fs = sf.read(path, dtype='float32')  

        

        play = """
        var audio= document.getElementById('audio');
        var playPauseBTN = document.getElementById('playPauseBTN');
        var count=0;
        function playPause(){
            if(count == 0){
                count = 1;
                audio.play();
                playPauseBTN.innerHTML = "Pause &#9208;"
            }else{
                count = 0;
                audio.pause();
                playPauseBTN.innerHTML = "Play &#9658;"
                }
            }
            """

        source = str(source)
        # save_img = not nosave and not source.endswith('.txt')  # save inference images
        # is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() #or source.endswith('.txt') or (is_url and not is_file)
        # if is_url and is_file:
        #     source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        # if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
        # else:
        #     dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        #     bs = 1  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        # global object
        # object = ''
        ctr = 0
        # object = ''
        for path, im, im0s, vid_cap, s in dataset:
            
            ctr += 1
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                object = ''
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        objects = ['Mute', 'Play', 'Stop', 'Volume-20', 'Volume-40', 'Volume-60', 'Volume-80', 'Volume-Max']
                        object = objects[int(cls.item())]
                        
                            

                            # print(ctr)

                            # # if ctr%100 == 0: 
                            # #     volume_40()
                            #     # ctr = 0  
                            # if object == "Play" and ctr > 100:
                            #     js2py.eval_js(play)
                            #     print("play")
                            #     ctr = 0  
                                # countdown(3)
                            # elif object == "Stop":
                            #     stop()
                            # elif object == "Mute":
                            #     Mute()  
                            # elif object == "Volume-20":
                            #     volume_20()  
                            # elif object == "Volume-40":
                            #     volume_40()  
                            # elif object == "Volume-60":
                            #     volume_60()  
                            # elif object == "Volume-80":
                            #     volume_80()              
                            # else:
                            #     volume_max()

                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            

                # Stream results
                im0 = annotator.result()
    
                yield cv2.imencode('.jpg', im0)[1].tobytes(), object

    
            
