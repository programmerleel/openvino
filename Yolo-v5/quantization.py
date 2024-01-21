# -*- coding: utf-8 -*-
# @Time    : 2024/01/18 11:05
# @Author  : LiShiHao
# @FileName: quantization.py
# @Software: PyCharm

from utils.general import download
from utils.datasets import create_dataloader
from utils.general import check_dataset
from export import attempt_load
from val import run as validation_fn

import argparse
import sys
import os

import openvino as ov
from openvino.tools.pot.api import DataLoader
import nncf
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.graph import load_model, save_model


# 继承yolo提供的dataloader，自定义扩展
def create_data_source(download_path,data_config_path):
    if not os.path.exists(os.path.join(download_path,"coco128")):
        urls = ["https://ultralytics.com/assets/coco128.zip"]
        download(urls, dir=download_path)

    data = check_dataset(data_config_path)
    data_source = create_dataloader(
        data["val"], imgsz=640, batch_size=1, stride=32, pad=0.5, workers=1
    )[0]

    return data_source

class POTDataloader(DataLoader):
    def __init__(self,data_source):
        super().__init__({})
        self.data_loader = data_source
        self.data_iter = iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader.dataset)

    def __getitem__(self, index):
        try:
            batch_data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch_data = next(self.data_iter)

        im, target, path, shape = batch_data

        im = im.float()
        im /= 255
        nb, _, height, width = im.shape
        img = im.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        annotation = dict()
        annotation["image_path"] = path
        annotation["target"] = target
        annotation["batch_size"] = nb
        annotation["shape"] = shape
        annotation["width"] = width
        annotation["height"] = height
        annotation["img"] = img

        return (index, annotation), img

def transform_fn(data_item):
    # unpack input images tensor
    images = data_item[0]
    # convert input tensor into float format
    images = images.float()
    # scale input
    images = images / 255
    # convert torch tensor to numpy array
    images = images.cpu().detach().numpy()
    return images

# Wrap framework-specific data source into the `nncf.Dataset` object.

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_path",type=str,default="./dataset")
    parser.add_argument("--data_config_path",type=str,default="./data/coco128.yaml")
    parser.add_argument("--model_name",type=str,default="yolov5s")
    parser.add_argument("--fp32_path",type=str,default="C:/Users/Administrator/Downloads/openvino-develop/Yolo-v5/model/yolov5s.xml")
    parser.add_argument("--model_path",type=str,default="C:/Users/Administrator/Downloads/openvino-develop/Yolo-v5/model")
    return parser.parse_args(argv)

# 不使用POT方式 可能因为版本原因 导致存在奇怪的错误 暂时没解决
def main(args):
    download_path = args.download_path
    data_config_path = args.data_config_path
    data_source = create_data_source(download_path,data_config_path)
    model_name = args.model_name
    fp32_path = args.fp32_path
    model_path = args.model_path

    nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)
    subset_size = 300
    preset = nncf.QuantizationPreset.MIXED

    core = ov.Core()
    ov_model = core.read_model(fp32_path)
    quantized_model = nncf.quantize(
        ov_model, nncf_calibration_dataset, preset=preset, subset_size=subset_size
    )
    nncf_int8_path = os.path.join(model_path,"nncf",model_name+"_int8.xml")
    ov.save_model(quantized_model, nncf_int8_path, compress_to_fp16=False)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

