import cv2
import h5py
from tqdm import tqdm
from skimage.io import imread
import os
import numpy as np
from top_k_alignment import crop_dataset, compute_channel_means, compute_patch_differences, top_k_patches
from top_k_alignment import apply_gamma_correction_to_patches, find_unselected_patches
from top_k_alignment import optimize_gamma, save_patches_to_folders, restore_images


def get_img_file(folder):
    """
    获取文件夹中的图像文件路径列表

    参数：
    - folder: 文件夹路径

    返回：
    - 图像文件路径列表
    """
    imagelist = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
            imagelist.append(os.path.join(folder, filename))
    return imagelist

def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def bgr_to_ycrcb(path):
    one = cv2.imread(path,1)
    one = one.astype('float32')
    (B, G, R) = cv2.split(one)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    return Y, cv2.merge([Cr,Cb])


if __name__ == '__main__':

    dataset1_folder = './dataset/RoadScene/'
    dataset1_train_folder = './dataset/RoadScene/train/'
    dataset2_folder = './dataset/M3FD_Fusion/'
    image_size = (64, 64)  # Adjust patch size as needed
    stride = 200

    # Crop datasets
    dataset1_visible_patches, dataset1_infrared_patches = crop_dataset(dataset1_folder, image_size, stride)
    dataset1_train_visible_patches, dataset1_train_infrared_patches = crop_dataset(dataset1_train_folder, image_size, stride)
    dataset2_visible_patches, dataset2_infrared_patches = crop_dataset(dataset2_folder, image_size, stride)

    # 输出两个数据集中包含的 patch 数量
    print("Dataset 1 contains", len(dataset1_visible_patches), "patches.")
    print("Dataset_train 1 contains", len(dataset1_train_visible_patches), "patches.")
    print("Dataset 2 contains", len(dataset2_visible_patches), "patches.")

    # Calculate channel means
    dataset1_channel_means = compute_channel_means(dataset1_visible_patches)
    dataset2_channel_means = compute_channel_means(dataset2_visible_patches)

    print("Dataset 1 channel means:", dataset1_channel_means)
    print("Dataset 2 channel means:", dataset2_channel_means)

    # 计算dataset2中每个图像块与目标数据集的像素均值差异度
    dataset2_differences = compute_patch_differences(dataset2_visible_patches, dataset1_channel_means)

    # # 选择差异度最小的top-k个图像块
    k = 1000  # 选择top-k数量，根据需要调整
    top_k_dataset2_visible_patches, top_k_dataset2_infrared_patches = top_k_patches(dataset2_visible_patches, dataset2_infrared_patches, dataset2_differences, k)
    top_k_dataset2_visible_patches_channel_means = compute_channel_means(top_k_dataset2_visible_patches)
    print('Top-k_patches channel means', top_k_dataset2_visible_patches_channel_means)

    # 优化gamma值
    best_gamma_values = optimize_gamma(top_k_dataset2_visible_patches, dataset1_channel_means)

    # 应用最佳的gamma值到图像块
    best_gamma_B, best_gamma_G, best_gamma_R = best_gamma_values
    top_k_gamma_corrected_patches = apply_gamma_correction_to_patches(top_k_dataset2_visible_patches, best_gamma_B, best_gamma_G, best_gamma_R)
    top_k_gamma_corrected_patches_channel_means = compute_channel_means(top_k_gamma_corrected_patches)
    print("Gamma_corrected_patches channel means:", top_k_gamma_corrected_patches_channel_means)

    # 输出最佳的gamma值
    print("Best gamma values:", best_gamma_values)

    merged_dataset_visible_patches = dataset1_train_visible_patches + top_k_gamma_corrected_patches
    merged_dataset_infrared_patches = dataset1_train_infrared_patches + top_k_dataset2_infrared_patches

    print('merged_dataset_visible_patches',len(merged_dataset_visible_patches))
    print('merged_dataset_infrared_patches',len(merged_dataset_infrared_patches))

    # 创建保存合并数据集的文件夹
    merged_dataset_folder = './dataset/data/top-k-ImPatch/merged/'
    if not os.path.exists(merged_dataset_folder):
        os.makedirs(merged_dataset_folder)

    # 保存各个类型的图像块到相应的文件夹中
    save_patches_to_folders(merged_dataset_visible_patches, "vis", merged_dataset_folder)
    save_patches_to_folders(merged_dataset_infrared_patches, "ir", merged_dataset_folder)

    data_name = "RSM3"

    IR_files = sorted(get_img_file('./dataset/data/top-k-ImPatch/merged/ir'))
    VIS_files = sorted(get_img_file('./dataset/data/top-k-ImPatch/merged/vis'))

    assert len(IR_files) == len(VIS_files)
    h5f = h5py.File(os.path.join('./dataset/data',
                                 data_name + '_imgsize_' + str(image_size[0]) + '_stride_' + str(stride) + '_k_' + str(k) + '.h5'),
                    'w')
    h5_ir = h5f.create_group('ir_patchs')
    h5_vis = h5f.create_group('vis_patchs')
    train_num = 0
    for i in tqdm(range(len(IR_files))):
        I_VIS = imread(VIS_files[i]).astype(np.float32).transpose(2, 0, 1) / 255.  # [3, H, W] Uint8->float32
        I_VIS = rgb2y(I_VIS)  # [1, H, W] Float32
        I_IR = cv2.imread(IR_files[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)[None, :, :] / 255.

        # 可以省略裁剪步骤，直接将图像块写入 HDF5 文件
        h5_ir.create_dataset(str(train_num), data=I_IR, dtype=I_IR.dtype, shape=I_IR.shape)
        h5_vis.create_dataset(str(train_num), data=I_VIS, dtype=I_VIS.dtype, shape=I_VIS.shape)
        train_num += 1

    h5f.close()

    with h5py.File(os.path.join('./dataset/data',
                                data_name + '_imgsize_' + str(image_size[0]) + '_stride_' + str(stride) + '_k_' + str(k) + '.h5'),
                   "r") as f:
        for key in f.keys():
            print(f[key], key, f[key].name)
