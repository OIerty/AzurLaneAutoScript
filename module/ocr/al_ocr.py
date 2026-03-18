import os
import numpy as np
import cv2
from PIL import Image

from module.exception import RequestHumanTakeover
from module.logger import logger

try:
    logger.info('Loading OCR dependencies')
    from cnocr import CnOcr
except Exception as e:
    logger.critical(f'Failed to load OCR dependencies: {e}')
    # Define dummy classes to prevent ImportErrors in other modules
    class CnOcr:
        pass


class AlOcr(CnOcr):
    def __init__(self, **kwargs):
        # Default to the user's requested model structure
        if 'rec_model_name' not in kwargs:
            kwargs['rec_model_name'] = 'densenet_lite_136-gru'
            
        if 'det_model_name' not in kwargs:
            kwargs['det_model_name'] = 'naive_det'
            
        # Provide the fp of the custom ONNX model if it exists
        model_fp = os.path.join('bin', 'cnocr_models', 'cnocr-v2.3-densenet_lite_136-gru-epoch=004-ft-model.onnx')
        if os.path.exists(model_fp) and 'rec_model_fp' not in kwargs:
            kwargs['rec_model_fp'] = model_fp
            
        # Filter out old mxnet parameters to avoid unexpected TypeError in new cnocr
        valid_keys = ['rec_model_name', 'det_model_name', 'cand_alphabet', 'context', 'rec_model_fp', 'rec_vocab_fp']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        
        self._kwargs = filtered_kwargs
        self._model_loaded = False
        
    def init(self):
        super().__init__(**self._kwargs)
        self._model_loaded = True

    def _ensure_loaded(self):
        if not self._model_loaded:
            self.init()

    def _extract_text(self, res):
        """
        The new cnocr returns dicts like {'text': 'abc', 'score': 0.99}.
        The old alas codebase expects structures compatible with ''.join() for single strings/lists.
        By returning just the 'text' value, we ensure compatibility.
        """
        if isinstance(res, dict) and 'text' in res:
            return res['text']
        elif isinstance(res, list):
            return [self._extract_text(r) for r in res]
        return res

    def ocr(self, img_fp):
        self._ensure_loaded()
        res = super().ocr(img_fp)
        return self._extract_text(res)

    def ocr_for_single_line(self, img_fp):
        self._ensure_loaded()
        res = super().ocr_for_single_line(img_fp)
        return self._extract_text(res)

    def ocr_for_single_lines(self, img_list):
        self._ensure_loaded()
        res = super().ocr_for_single_lines(img_list)
        return self._extract_text(res)

    def set_cand_alphabet(self, cand_alphabet):
        # CnOcr v2 does not have dynamic set_cand_alphabet anymore.
        # Characters are handled during init or implicitly well enough by the new model.
        pass

    def atomic_ocr(self, img_fp, cand_alphabet=None):
        self._ensure_loaded()
        return self.ocr(img_fp)

    def atomic_ocr_for_single_line(self, img_fp, cand_alphabet=None):
        self._ensure_loaded()
        return self.ocr_for_single_line(img_fp)

    def atomic_ocr_for_single_lines(self, img_list, cand_alphabet=None):
        self._ensure_loaded()
        return self.ocr_for_single_lines(img_list)

    def debug(self, img_list):
        """
        Visual debugging of images fed to OCR.
        """
        if len(img_list) > 0:
            # Ensure images are properly formatted for hconcat
            concat_list = []
            for img in img_list:
                if len(img.shape) == 2:
                    # Gray to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                concat_list.append(img)
                
            image = cv2.hconcat(concat_list)
            Image.fromarray(image).show()
