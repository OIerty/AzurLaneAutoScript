from module.base.decorator import cached_property


class OcrModel:
    @cached_property
    def azur_lane(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr(
            rec_model_name='scene-densenet_lite_136-gru',
            rec_model_fp='bin/cnocr_models/cnocr-v2.3-scene-densenet_lite_136-gru-epoch=004-ft-model.onnx',
            name='azur_lane'
        )

    @cached_property
    def azur_lane_jp(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr()

    @cached_property
    def cnocr(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr()

    @cached_property
    def jp(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr()

    @cached_property
    def tw(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr()


OCR_MODEL = OcrModel()
