# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import time
import shutil
import platform
import numpy as np
from datetime import datetime

#import torch
import paddle
#import torchvision as tv
import paddle.vision as pv
#import torch.backends.cudnn as cudnn

# from torch.utils.tensorboard import SummaryWriter

import yaml
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
#import torchvision.utils as vutils

##---------------------------组合实现save_image---------------------------------------------------##
import pathlib
import warnings
import math
from PIL import Image
from typing import Union, Optional, List, Tuple, Text, BinaryIO

@paddle.no_grad()
def make_grid(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
              nrow: int=8,
              padding: int=2,
              normalize: bool=False,
              value_range: Optional[Tuple[int, int]]=None,
              scale_each: bool=False,
              pad_value: int=0,
              **kwargs) -> paddle.Tensor:
    if not (isinstance(tensor, paddle.Tensor) or
            (isinstance(tensor, list) and all(
                isinstance(t, paddle.Tensor) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding,
                        width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(
                x + 1) * width] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str]=None,
               **kwargs) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose(
        [1, 2, 0]).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

##-----------------------------------------------------------------------------------------------##

##----------------------组合实现ToPILImage-----------------------------------------------##
import paddle
import PIL
import numbers
import numpy as np
from PIL import Image
from paddle.vision.transforms import BaseTransform
from paddle.vision.transforms import functional as F


class ToPILImage(BaseTransform):
    def __init__(self, mode=None, keys=None):
        super(ToTensor, self).__init__(keys)

    def _apply_image(self, pic):
        #mode = self.mode
        """
        Args:
            pic (Tensor|np.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL: Converted image.
        """
        if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(
                type(pic)))

        elif isinstance(pic, paddle.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

        npimg = pic
        if isinstance(pic, paddle.Tensor) and "float" in str(pic.numpy(
        ).dtype) and mode != 'F':
            pic = pic.mul(255).byte()
        if isinstance(pic, paddle.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError(
                'Input pic must be a paddle.Tensor or NumPy ndarray, ' +
                'not {}'.format(type(npimg)))

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            if mode is not None and mode != expected_mode:
                raise ValueError(
                    "Incorrect mode ({}) supplied for input type {}. Should be {}"
                    .format(mode, np.dtype, expected_mode))
            mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if mode is not None and mode not in permitted_2_channel_modes:
                raise ValueError("Only modes {} are supported for 2D inputs".
                                 format(permitted_2_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if mode is not None and mode not in permitted_4_channel_modes:
                raise ValueError("Only modes {} are supported for 4D inputs".
                                 format(permitted_4_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if mode is not None and mode not in permitted_3_channel_modes:
                raise ValueError("Only modes {} are supported for 3D inputs".
                                 format(permitted_3_channel_modes))
            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGB'

        if mode is None:
            raise TypeError('Input type {} is not supported'.format(
                npimg.dtype))

        return Image.fromarray(npimg, mode=mode)
##----------------------组合实现ToPILImage---------------------------------------------------------##

##### option parsing ######
def print_options(config_dict):
    print("------------ Options -------------")
    for k, v in sorted(config_dict.items()):
        print("%s: %s" % (str(k), str(v)))
    print("-------------- End ----------------")


def save_options(config_dict):
    from time import gmtime, strftime

    file_dir = os.path.join(config_dict["checkpoint_dir"], config_dict["name"])
    mkdir_if_not(file_dir)
    file_name = os.path.join(file_dir, "opt.txt")
    with open(file_name, "wt") as opt_file:
        opt_file.write(os.path.basename(sys.argv[0]) + " " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
        opt_file.write("------------ Options -------------\n")
        for k, v in sorted(config_dict.items()):
            opt_file.write("%s: %s\n" % (str(k), str(v)))
        opt_file.write("-------------- End ----------------\n")


def config_parse(config_file, options, save=True):
    with open(config_file, "r") as stream:
        config_dict = yaml.safe_load(stream)
        config = edict(config_dict)

    for option_key, option_value in vars(options).items():
        config_dict[option_key] = option_value
        config[option_key] = option_value

    if config.debug_mode:
        config_dict["num_workers"] = 0
        config.num_workers = 0
        config.batch_size = 2
        if isinstance(config.gpu_ids, str):
            config.gpu_ids = [int(x) for x in config.gpu_ids.split(",")][0]

    print_options(config_dict)
    if save:
        save_options(config_dict)

    return config


###### utility ######
def to_np(x):
    return x.cpu().numpy()


def prepare_device(use_gpu, gpu_ids):
    if use_gpu:
        cudnn.benchmark = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if isinstance(gpu_ids, str):
            gpu_ids = [int(x) for x in gpu_ids.split(",")]
            paddle.device.set_device(gpu_ids[0])
            device = paddle.device("cuda:" + str(gpu_ids[0]))
        else:
            paddle.device.set_device(gpu_ids)
            device = paddle.device("cuda:" + str(gpu_ids))
        print("running on GPU {}".format(gpu_ids))
    else:
        device = paddle.device("cpu")
        print("running on CPU")

    return device


###### file system ######
def get_dir_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def mkdir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


##### System related ######
class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        elapse = time.time() - self.start_time
        print(self.msg % elapse)


###### interactive ######
def get_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def clean_tensorboard(directory):
    tensorboard_list = os.listdir(directory)
    SIZE_THRESH = 100000
    for tensorboard in tensorboard_list:
        tensorboard = os.path.join(directory, tensorboard)
        if get_size(tensorboard) < SIZE_THRESH:
            print("deleting the empty tensorboard: ", tensorboard)
            #
            if os.path.isdir(tensorboard):
                shutil.rmtree(tensorboard)
            else:
                os.remove(tensorboard)


def prepare_tensorboard(config, experiment_name=datetime.now().strftime("%Y-%m-%d %H-%M-%S")):
    tensorboard_directory = os.path.join(config.checkpoint_dir, config.name, "tensorboard_logs")
    mkdir_if_not(tensorboard_directory)
    clean_tensorboard(tensorboard_directory)
    tb_writer = SummaryWriter(os.path.join(tensorboard_directory, experiment_name), flush_secs=10)

    # try:
    #     shutil.copy('outputs/opt.txt', tensorboard_directory)
    # except:
    #     print('cannot find file opt.txt')
    return tb_writer


def tb_loss_logger(tb_writer, iter_index, loss_logger):
    for tag, value in loss_logger.items():
        tb_writer.add_scalar(tag, scalar_value=value.item(), global_step=iter_index)


def tb_image_logger(tb_writer, iter_index, images_info, config):
    ### Save and write the output into the tensorboard
    tb_logger_path = os.path.join(config.output_dir, config.name, config.train_mode)
    mkdir_if_not(tb_logger_path)
    for tag, image in images_info.items():
        if tag == "test_image_prediction" or tag == "image_prediction":
            continue
        image = make_grid(image.cpu())
        image = paddle.clip(image, 0, 1)
        tb_writer.add_image(tag, img_tensor=image, global_step=iter_index)
        '''
        tv.transforms.functional.to_pil_image(image).save(
            os.path.join(tb_logger_path, "{:06d}_{}.jpg".format(iter_index, tag))
        )
        '''
        ToPILImage()(image).save(
            os.path.join(tb_logger_path, "{:06d}_{}.jpg".format(iter_index, tag))
        )



def tb_image_logger_test(epoch, iter, images_info, config):

    url = os.path.join(config.output_dir, config.name, config.train_mode, "val_" + str(epoch))
    if not os.path.exists(url):
        os.makedirs(url)
    scratch_img = images_info["test_scratch_image"].data.cpu()
    if config.norm_input:
        scratch_img = (scratch_img + 1.0) / 2.0
    scratch_img = paddle.clip(scratch_img, 0, 1)
    gt_mask = images_info["test_mask_image"].data.cpu()
    predict_mask = images_info["test_scratch_prediction"].data.cpu()

    predict_hard_mask = (predict_mask.data.cpu() >= 0.5).float()

    imgs = paddle.conconcat((scratch_img, predict_hard_mask, gt_mask), 0)
    img_grid = save_image(
        imgs, os.path.join(url, str(iter) + ".jpg"), nrow=len(scratch_img), padding=0, normalize=True
    )


def imshow(input_image, title=None, to_numpy=False):
    inp = input_image
    if to_numpy or type(input_image) is paddle.Tensor:
        inp = input_image.numpy()

    fig = plt.figure()
    if inp.ndim == 2:
        fig = plt.imshow(inp, cmap="gray", clim=[0, 255])
    else:
        fig = plt.imshow(np.transpose(inp, [1, 2, 0]).astype(np.uint8))
    plt.axis("off")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)


###### vgg preprocessing ######
def vgg_preprocess(tensor):
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = paddle.concat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - paddle.to_tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(
        1, 3, 1, 1
    )
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst


def torch_vgg_preprocess(tensor):
    # pytorch version normalization
    # note that both input and output are RGB tensors;
    # input and output ranges in [0,1]
    # normalize the tensor with mean and variance
    tensor_mc = tensor - paddle.to_tensor([0.485, 0.456, 0.406]).type_as(tensor).view(1, 3, 1, 1)
    tensor_mc_norm = tensor_mc / paddle.to_tensor([0.229, 0.224, 0.225]).type_as(tensor_mc).view(1, 3, 1, 1)
    return tensor_mc_norm


def network_gradient(net, gradient_on=True):
    if gradient_on:
        for param in net.parameters():
            param.requires_grad = True
    else:
        for param in net.parameters():
            param.requires_grad = False
    return net
