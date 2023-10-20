import argparse
import logging
import numpy as np
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
    模型转换的时候 主流的 pytorch tensorflow paddlepadlle可以直接转换ir
    但是为了保证对不同格式的通用性，通过onnx作为媒介
    后面会跟上通过onnx转换rknn，tensorrt
'''

def create_log(log_path):
    logging.basicConfig(filename=log_path,level=logging.INFO,format='%(asctime)s - %(name)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s')

# 加载权重有两种方式，根据保存的方式自行调整
def load_pytorch_model(input_model_path):
    model = torch.load(input_model_path)
    # model = net()
    # model.load_state_dict(torch.load(input_model_path))
    return model

def pytorch_export_onnx(model,onnx_model_path):
    onnx_model_path = Path(onnx_model_path)
    if not os.path.exists(onnx_model_path.parent):
        os.makedirs(onnx_model_path.parent)
    
    model.eval()
    input = torch.randn((1,3,640,640),device='cuda')
    fp = False   # 是否执行fp16操作
    simplify = True # 是否简化网络结构方便查看
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
            model_onnx = float16_converter.convert_float_to_float16(
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
        session = onnxruntime.InferenceSession(onnx_model_path)
        onnx_input = input.detach().cpu().numpy() if input.requires_grad else input.cpu().numpy()
        onnx_output = session.run(None,{"images": onnx_input})

        with torch.no_grad():
            output = model(input)
            output = output.detach().cpu().numpy()
        np.testing.assert_almost_equal(output, onnx_output[0], decimal=4)

    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

def onnx_export_ir(onnx_model_path,ir_model_path):
    '''
    最新的api找那个显示，在进行模型转换的时候会自动的进行fp16压缩
    如果不需要的话，可以在save_model中显式的设置compress_to_fp16为False
    但是在实际的测试过程中，使用onnx fp32模型进行转换，并不会自动进行压缩

    在使用best.pt直接转换ir模型的过程中，在loading model的过程中报错
    AttributeError: 'torch._C.TensorType' object has no attribute 'dtype'
    但是在使用pt->onnx->ir是可行的，这里暂时不知道问题在哪个位置
    '''
    # 转换模型
    try:
        model_ov = ov.convert_model(onnx_model_path)
        ov.save_model(model_ov,ir_model_path)
    # 精度验证
        input = np.random.randn(1,3,640,640).astype(np.float32)
        session = onnxruntime.InferenceSession(onnx_model_path)
        onnx_input = input
        onnx_output = session.run(None,{"images": onnx_input})

        core = ov.Core()
        read_model = core.read_model(ir_model_path)
        compile_model = core.compile_model(read_model,'CPU')
        # input_layer = compile_model.input(0)
        output_layer = compile_model.output(0)
        result = compile_model(input)[output_layer]

        np.testing.assert_almost_equal(onnx_output[0], result, decimal=4)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc)


def main(args):
    input_model_path = args.input_model_path
    onnx_model_path = args.onnx_model_path
    ir_model_path = args.ir_model_path
    log_path = args.log_path
    create_log(log_path)
    model = load_pytorch_model(input_model_path)
    # pytorch_export_onnx(model,onnx_model_path)
    onnx_export_ir(onnx_model_path,ir_model_path)

# 导出的参数暂时在上面函数中调整
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path',type=str,help='',default='/home/code/ResNet18/weights/best.pt')
    parser.add_argument('--onnx_model_path',type=str,help='',default='/home/code/ResNet18/weights/best.onnx')
    parser.add_argument('--ir_model_path',type=str,help='',default='/home/code/ResNet18/weights/best.xml')
    parser.add_argument('--log_path',type=str,help='',default='/home/code/openvino/logs.txt')
    return parser.parse_args(argv)

if __name__=='__main__':
    main(parse_arguments(sys.argv[1:]))