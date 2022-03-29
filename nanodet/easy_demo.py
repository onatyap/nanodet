import os
import cv2
import torch


device = torch.device('cpu')

from nanodet.util import cfg, load_config, Logger

config_path = 'config/nanodet-plus-m_416.yml'
model_path = 'model/nanodet-plus-m_416_checkpoint.ckpt'
image_path = 'demo_mnn/imgs/000252.jpg'

load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)

from demo.demo import Predictor

predictor = Predictor(cfg, model_path, logger, device=device)

meta, res = predictor.inference(image_path)

from nanodet.util import overlay_bbox_cv

from IPython.display import display
from PIL import Image

def cv2_imshow(a, convert_bgr_to_rgb=True, write_image=True, write_path=None):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
            (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
            image.
        convert_bgr_to_rgb: switch to convert BGR to RGB channel.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if convert_bgr_to_rgb and a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))
    if write_image:
      if write_path is None:
        os.makedirs('results')
        write_path = 'results/result.png'
      
      cv2.imwrite(write_path)
      
      
In [10]:
result = overlay_bbox_cv(meta['raw_img'], res[0], cfg.class_names, score_thresh=0.35)
In [11]:
imshow_scale = 1.0
cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))
