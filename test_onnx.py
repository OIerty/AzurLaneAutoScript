import onnxruntime as ort

model_fp = 'bin/cnocr_models/cnocr-v2.3-number-densenet_lite_136-fc-epoch=023.onnx'
session = ort.InferenceSession(model_fp)
outputs = session.get_outputs()
for o in outputs:
    print(o.name, o.shape)
