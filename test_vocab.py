from cnocr import CnOcr

# Try to initialize with number-densenet_lite_136-fc to see if it implicitly loads number vocab
try:
    ocr = CnOcr(rec_model_name='number-densenet_lite_136-fc', rec_model_fp='bin/cnocr_models/cnocr-v2.3-number-densenet_lite_136-fc-epoch=023.onnx', det_model_name='naive_det')
    print("number-densenet_lite_136-fc success!")
    print("vocab size:", len(ocr.rec_model._vocab))
    print("vocab preview:", ocr.rec_model._vocab[:10])
except Exception as e:
    print("number-densenet_lite_136-fc Error:", e)

