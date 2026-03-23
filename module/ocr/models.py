from module.base.decorator import cached_property


class OcrModel:
    @cached_property
    def azur_lane(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr(name='en')

    @cached_property
    def azur_lane_jp(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr(name='en')

    @cached_property
    def cnocr(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr(name='zhcn')

    @cached_property
    def jp(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr(name='en')

    @cached_property
    def tw(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr(name='zhcn')

    def unload(self):
        """卸载所有已缓存的 OCR 模型以释放内存"""
        for attr in ['azur_lane', 'azur_lane_jp', 'cnocr', 'jp', 'tw']:
            if attr in self.__dict__:
                self.__dict__[attr].unload()
                del self.__dict__[attr]

OCR_MODEL = OcrModel()

