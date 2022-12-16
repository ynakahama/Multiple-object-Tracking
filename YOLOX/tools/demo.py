#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import sys
sys.path.append(os.getcwd())
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

#motpyでのトラッキング
from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_track

class MOT:
    def __init__(self):
        self.tracker = MultiObjectTracker(dt=0.1)

    def track(self, outputs, ratio):
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            outputs = [Detection(box=box[:4] / ratio, score=box[4] * box[5], class_id=box[6]) for box in outputs]
        else:
            outputs = []

        self.tracker.step(detections=outputs)
        tracks = self.tracker.active_tracks()
        return tracks




IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        #print("img",img)
        #print("vis_res",vis_res)
        #print("self.cls_names",self.cls_names)#要素のリスト
        #print("cls",cls)#要素リストの番号指定
        #print("bboxes",bboxes)
        #print("scores",scores)
        #print("cls_conf",cls_conf)
        return vis_res,scores


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    print("START imageflow")
    mot = MOT()
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)                      
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.splitext(args.path.split("/")[-1])[0] + '.mp4')
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    path_num=0
    data_dic4 = {}
    data_dic5 = {}
    data_dic6 = {}
    data_dic7 = {}
    data_dic8 = {}
    data100_dic = {}
    all_data_dic = {}
    while True:
        #print("imageflow while")
        ret_val, frame = cap.read()
        #cv2.imwrite('pic/k.jpg', frame)
        #print("ret_val",ret_val)
        #print("frame",frame)
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            #print("outputs",outputs)
            #print("img_info",img_info['raw_img'][0][0])
            #print("predictor.confthre",predictor)
            result_frame1,scores = predictor.visual(outputs[0], img_info, predictor.confthre)
            result_frame = frame
            tracks = mot.track(outputs, img_info['ratio'])
            #print("tracks",tracks)
            for trc in tracks:
                #print("result_frame",result_frame[0][0][0])
                draw_track(result_frame, trc, thickness=1)#バウンディングボックスの表示
            if args.save_result:#セーブ用
                vid_writer.write(result_frame)
            else:#出力用
                #print("result_frame",result_frame)
                cv2.imshow('frame', result_frame)
                cv2.imwrite('pic/image/%06d.jpg' % path_num, result_frame)
                path_w = 'pic/txt/%06d.txt' %path_num#外部ファイルへMotの情報を書き出し
                with open(path_w, mode='w') as f:
                    for trc in tracks:
                        #center_x, center_y, width_x, width_y
                        f.write(str(trc[3])+" "+str((trc[1][0]+trc[1][2])/2)+" "+str((trc[1][1]+trc[1][3])/2)+" "+str(trc[1][2]-trc[1][0])+" "+str(trc[1][3]+trc[1][1]))
                        f.write("\n")
                        #print("trc:",trc)
                        #print("trc0:",trc[0])
                        #print("trc1:",trc[1])
                        #print("trc2:",trc[2])
                        #print("trc3:",trc[3])

                        #data100_dic.update({path_num+trc[0]: trc[0]})#同じpath_numで更新されてしまう
                        #if (path_num - 100) in data100_dic:


                        
                        #if trc[2] > 0.4:
                        #    data_dic4.update({trc[0]: trc[3]})
                        #if trc[2] > 0.5:
                        #    data_dic5.update({trc[0]: trc[3]})
                        #if trc[2] > 0.6:
                        #    data_dic6.update({trc[0]: trc[3]})
                        #if trc[2] > 0.7:
                        #    data_dic7.update({trc[0]: trc[3]})
                        #if trc[2] > 0.8:
                        #    data_dic8.update({trc[0]: trc[3]})
                        #if trc[0] in all_data_dic:
                        #    if all_data_dic[trc[0]]>trc[2]:
                        #        all_data_dic.update({trc[0]: trc[3]})
                        #else:
                        #    all_data_dic.update({trc[0]: trc[3]})
                f.close()
                #print("len",len(tracks),len(scores))
                #print("scores",scores)
                path_num = 1 + path_num
                print(path_num)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
        else:
            break
    print("data_dic8:",data_dic8)
    print("all_data_dic:",all_data_dic)
    print("data4_len",len(data_dic4))
    print("data5_len",len(data_dic5))
    print("data6_len",len(data_dic6))
    print("data7_len",len(data_dic7))
    print("data8_len",len(data_dic8))
    print("all_len",len(all_data_dic))


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    #print(COCO_CLASSES)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
