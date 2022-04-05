import os
import cv2
import torch
import click

import yaml
from yaml import Loader, Dumper
from demo.demo import Predictor
from nanodet.util import cfg, load_config, Logger


@click.group(help="Object Detection Command Line")
@click.pass_context
def cli(ctx):
    pass

@cli.command(help='Run inference on input images')
@click.option('--image_path', default='/data', help='Train configuration file')
@click.option('--config_path', default='config/nanodet-plus-m_416.yml', help='Dataset configuration file')
@click.option('--model_path', default='model/nanodet-plus-m_416_checkpoint.ckpt', help='Model configuration file')
def inference(image_path, config_path, model_path):
    device = torch.device('cpu')
    load_config(cfg, config_path)
    logger = Logger(-1, use_tensorboard=False)

    predictor = Predictor(cfg, model_path, logger, device=device)
    
    if isinstance(image_path, str) and '.' in image_path:
        image_paths = [image_path]
    else:
        image_paths = os.listdir(image_path)

    
    for i, im in enumerate(image_paths):
        
        _, res = predictor.inference(image_path+'/'+im)
        
        with open(f'/output/result_{i}.yml', 'w') as out:
            yaml.dump(res[0], out)
    print('complete')

if __name__ == '__main__':
    cli()
