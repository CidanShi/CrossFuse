import numpy as np
import cv2
import os
import torch
from skimage import img_as_ubyte
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    # imsave(os.path.join(savepath, "{}.png".format(imagename)),image)
    imsave(os.path.join(savepath, "{}.png".format(imagename)), image.astype(np.uint8))

# def img_save(image, imagename, savepath):
#     if not os.path.exists(savepath):
#         os.makedirs(savepath)
#
#     # 获取原始文件名和后缀
#     name, extension = os.path.splitext(imagename)
#
#     # 拼接保存路径时使用原始后缀
#     save_filename = "{}{}".format(name, extension)
#
#     # 保存图像
#     imsave(os.path.join(savepath, save_filename), image.astype(np.uint8))



def rgb_to_ycbcr(rgb_image):
    # 将 RGB 图像的形状转换为 [batch_size, height * width, 3]
    rgb_image_flattened = rgb_image.view(rgb_image.size(0), 3, -1).transpose(1, 2).contiguous()

    # 定义 YCbCr 转换矩阵
    transform_matrix = torch.tensor([
        [0.257, 0.564, 0.098],
        [-0.148, -0.291, 0.439],
        [0.439, -0.368, -0.071]
    ], dtype=torch.float32).to(rgb_image.device)

    # 执行 YCbCr 转换
    ycbcr_image_flattened = torch.matmul(rgb_image_flattened, transform_matrix.T)

    # 将 YCbCr 图像的形状还原为 [batch_size, 3, height, width]
    ycbcr_image = ycbcr_image_flattened.transpose(1, 2).view(rgb_image.size())

    return ycbcr_image
