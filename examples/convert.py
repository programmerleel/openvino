import argparse
import logging
import onnx
from onnxmltools.utils import float16_converter
import onnxruntime
import onnxsim
import openvino as ov
import os
from pathlib import Path
import sys
import torch
import torch._C as _C
import traceback

'''
模型转换：
    1.pytorch模型转onnx，onnx模型转ir
        这里如果是yolo的模型需要按照pytorch->onnx->ir，直接在yolo文件里转换onnx
    2.pytorch模型转换ir
'''

def create_log(log_path):
    logging.basicConfig(filename=log_path,level=logging.INFO,format='%(asctime)s - %(name)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s')

def export_onnx(input_model_path,onnx_model_path):
    onnx_model_path = Path(onnx_model_path)
    if not os.path.exists(onnx_model_path.parent):
        os.makedirs(onnx_model_path.parent)
    
    # 加载权重有两种方式，根据保存的方式自行调整
    model = torch.load(input_model_path)
    model.eval()
    input = torch.randn((1,3,640,640),device='cuda')
    fp = True
    simplify = True
    OperatorExportTypes = _C._onnx.OperatorExportTypes
    TrainingMode = _C._onnx.TrainingMode

    try:
        torch.onnx.export(
            model=model, # 需要导出的模型
            args=input,    # 模型的输入
            f=onnx_model_path, # 模型输出位置
            export_params=True,    # 导出模型是否带有参数
            verbose=False, # 是否在导出过程中打印日志
            training=TrainingMode.EVAL, # 模型训练、推理模式
            input_names=['images'],   # 列表 顺序分配给输入节点
            output_names=['output'],  # 列表 顺序分配给输出节点
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,    # 是否自定义算子
            opset_version=10,    # 算子库版本
            do_constant_folding=True, # 常量折叠
            dynamic_axes={'images': {0: 'batch'},
                          'output': {0: 'batch'}
                        }   # 动态batch
        )

        model_onnx = onnx.load(onnx_model_path) # 加载模型

        if fp:
            float16_converter.convert_float_to_float16(
                model_onnx,
                keep_io_types=True
                )   # 转fp16模型

        onnx.checker.check_model(model_onnx)    # 检查模型

        if simplify:
            model_onnx, check = onnxsim.simplify(
                model_onnx
                )   # 简化模型
            assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model_onnx, onnx_model_path)

        # 精度验证

    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

def export_ir(input_model_path,ir_model_path):
    model_ov = ov.convert_model(input_model_path)
    ov.save_model(model_ov,ir_model_path)


def main(args):
    input_model_path = args.input_model_path
    onnx_model_path = args.onnx_model_path
    ir_model_path = args.ir_model_path
    log_path = args.log_path
    create_log(log_path)
    export_onnx(input_model_path,onnx_model_path)
    # export_ir(input_model_path,ir_model_path)

# 一些参数在上面函数中调整
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path',type=str,help='',default='/home/code/openvino/models/best.pt')
    parser.add_argument('--onnx_model_path',type=str,help='',default='/home/code/openvino/models/best.onnx')
    parser.add_argument('--ir_model_path',type=str,help='',default='/home/code/openvino/models/best.xml')
    parser.add_argument('--log_path',type=str,help='',default='/home/code/openvino/logs.txt')
    return parser.parse_args(argv)

if __name__=='__main__':
    main(parse_arguments(sys.argv[1:]))