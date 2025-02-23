import os
import cv2
import numpy as np
from scipy.optimize import minimize
import re


# def crop_image(visible_image, infrared_image, patch_size, original_image_name):
#     """
#     将visible和infrared图像裁剪为指定大小的图像块（patches），并为每个图像块命名
#
#     参数：
#     - visible_image：visible图像
#     - infrared_image：infrared图像
#     - patch_size：裁剪后的图像块大小，格式为(width, height)
#     - original_image_name: 原始图像的文件名
#
#     返回：
#     - 裁剪后的图像块列表，每个元素是包含文件名和图像块的元组
#     """
#     height, width = visible_image.shape[:2]
#     patch_width, patch_height = patch_size
#     visible_patches = []
#     infrared_patches = []
#     for y in range(0, height, patch_height):
#         for x in range(0, width, patch_width):
#             visible_patch = visible_image[y:y + patch_height, x:x + patch_width]
#             infrared_patch = infrared_image[y:y + patch_height, x:x + patch_width]
#             name, extension = os.path.splitext(original_image_name)
#             # 构建图像块的文件名
#             patch_filename = f"{name}_patch_{x}_{y}{extension}"
#             visible_patches.append((patch_filename, visible_patch))
#             infrared_patches.append((patch_filename, infrared_patch))
#     return visible_patches, infrared_patches


def crop_images(visible_image, infrared_image, patch_size, stride, original_image_name):
    """
    将visible和infrared图像裁剪为指定大小的图像块（patches），并为每个图像块命名

    参数：
    - visible_image：visible图像
    - infrared_image：infrared图像
    - patch_size：裁剪后的图像块大小，格式为(width, height)
    - original_image_name: 原始图像的文件名

    返回：
    - 裁剪后的图像块列表，每个元素是包含文件名和图像块的元组
    """
    height, width = visible_image.shape[:2]
    patch_width, patch_height = patch_size

    visible_patches = []
    infrared_patches = []

    # stride_x = patch_width // 2
    # stride_y = patch_height // 2
    stride_x = 200
    stride_y = 200

    for y in range(0, height - patch_height + 1, stride):
        for x in range(0, width - patch_width + 1, stride):
            visible_patch = visible_image[y:y + patch_height, x:x + patch_width]
            infrared_patch = infrared_image[y:y + patch_height, x:x + patch_width]
            name, extension = os.path.splitext(original_image_name)
            # 构建图像块的文件名
            patch_filename = f"{name}_patch_{x}_{y}{extension}"
            visible_patches.append((patch_filename, visible_patch))
            infrared_patches.append((patch_filename, infrared_patch))
    return visible_patches, infrared_patches


def crop_dataset(dataset_folder, patch_size, stride):
    """
    将数据集中的 visible 和 infrared 图像裁剪为指定大小的图像块（patches）

    参数：
    - dataset_folder：数据集所在的文件夹路径
    - patch_size：裁剪后的图像块大小，格式为(width, height)

    返回：
    - 裁剪后的图像块列表，每个元素是包含文件名和 visible 图像块、infrared 图像块的元组
    """
    visible_patches = []
    infrared_patches = []
    visible_folder = os.path.join(dataset_folder, "visible")
    infrared_folder = os.path.join(dataset_folder, "infrared")
    for filename in os.listdir(visible_folder):
        visible_image_path = os.path.join(visible_folder, filename)
        infrared_image_path = os.path.join(infrared_folder, filename)
        visible_image = cv2.imread(visible_image_path)
        infrared_image = cv2.imread(infrared_image_path)
        if visible_image is not None and infrared_image is not None:
            visible_image_patches, infrared_image_patches = crop_images(visible_image, infrared_image, patch_size, stride, filename)

            visible_patches.extend(visible_image_patches)
            infrared_patches.extend(infrared_image_patches)
    return visible_patches, infrared_patches


def compute_channel_means(patches):
    """
    计算图像块列表中每个通道的像素均值

    参数：
    - patches：图像块列表

    返回：
    - 一个包含每个通道像素均值的列表
    """
    num_patches = len(patches)
    channel_sums = [0, 0, 0]  # 初始化每个通道的像素值总和

    # 遍历图像块列表，累加每个通道上的像素值
    for filename, patch in patches:
        for i in range(3):  # 三个通道
            channel_sums[i] += np.mean(patch[:, :, i])

    # 计算每个通道的像素均值
    channel_means = [channel_sum / num_patches for channel_sum in channel_sums]
    return channel_means


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def apply_gamma_correction_to_patches(patches, gamma_B, gamma_G, gamma_R):
    """
    对图像块列表中的每个图像块应用RGB三通道上的gamma校正

    参数：
    - patches：图像块列表
    - gamma_B：蓝色通道的gamma值
    - gamma_G：绿色通道的gamma值
    - gamma_R：红色通道的gamma值

    返回：
    - 经过gamma校正后的图像块列表
    """
    corrected_patches = []
    for filename, patch in patches:
        # 对每个通道应用gamma校正
        gammaImg_B = gammaCorrection(patch[:, :, 0], gamma_B)
        gammaImg_G = gammaCorrection(patch[:, :, 1], gamma_G)
        gammaImg_R = gammaCorrection(patch[:, :, 2], gamma_R)
        # 将三个通道合并成一个图像块
        gammaImg = np.dstack((gammaImg_B, gammaImg_G, gammaImg_R))
        corrected_patches.append((filename, gammaImg))
    return corrected_patches


def compute_patch_differences(dataset_patches, target_channel_means):
    """
    计算每个图像块与目标数据集之间的像素均值差异度

    参数：
    - dataset_patches：数据集中的图像块列表
    - target_channel_means：目标数据集在每个通道上的像素均值

    返回：
    - 每个图像块的总差异度列表
    """
    patch_differences = []
    for filaname, patch in dataset_patches:
        # 计算每个通道上的像素均值
        patch_channel_means = [np.mean(patch[:, :, i]) for i in range(3)]
        # 计算每个通道上的像素均值差异度
        channel_differences = [abs(patch_mean - target_mean) for patch_mean, target_mean in
                               zip(patch_channel_means, target_channel_means)]
        # 计算总差异度
        total_difference = sum(channel_differences)

        patch_differences.append(total_difference)
    return patch_differences


def top_k_patches(visible_patches, infrared_patches, differences, k):
    """
    选择差异度最小的 k 个 visible 和 infrared patches

    参数：
    - visible_patches：visible 图像块列表
    - infrared_patches：infrared 图像块列表
    - differences：图像块的差异度列表
    - k：要选择的图像块数量

    返回：
    - 差异度最小的 k 个 visible 和 infrared patches 的元组列表
    """
    top_k_indices = sorted(range(len(differences)), key=lambda i: differences[i])[:k]
    top_k_visible_patches = [visible_patches[i] for i in top_k_indices]
    top_k_infrared_patches = [infrared_patches[i] for i in top_k_indices]
    return top_k_visible_patches, top_k_infrared_patches



def find_unselected_patches(all_patches, selected_patches):
    """
    找到未被选择的图像块

    参数：
    - all_patches：所有图像块的列表，每个元素为(filename, patch)元组
    - selected_patches：被选择的图像块的列表，每个元素为(filename, patch)元组

    返回：
    - 未被选择的图像块的列表，每个元素为(filename, patch)元组
    """
    # 获取被选择的图像块的文件名集合
    selected_patch_filenames = {filename for filename, _ in selected_patches}

    unselected_patches = []
    for filename, patch in all_patches:
        # 如果当前图像块的文件名不在被选择的文件名集合中，则将其添加到未选择的图像块列表中
        if filename not in selected_patch_filenames:
            unselected_patches.append((filename, patch))

    return unselected_patches


def optimize_gamma(visible_patches, target_channel_means):
    """
    优化gamma值，使得经过gamma校正后的 visible 图像块与目标数据集的像素均值尽可能接近

    参数：
    - visible_patches：visible 图像块列表
    - target_channel_means：目标数据集在每个通道上的像素均值

    返回：
    - 最佳的gamma值列表
    """

    def loss_function(gamma_values, visible_patches, target_channel_means):
        """
        损失函数：用于优化gamma值，使得经过gamma校正后的 visible 图像块与目标数据集的像素均值尽可能接近

        参数：
        - gamma_values：包含三个通道的gamma值的列表
        - visible_patches：visible 图像块列表
        - target_channel_means：目标数据集在每个通道上的像素均值

        返回：
        - 损失值
        """
        gamma_B, gamma_G, gamma_R = gamma_values
        corrected_visible_patches = apply_gamma_correction_to_patches(visible_patches, gamma_B, gamma_G, gamma_R)

        # 计算经过gamma校正后的 visible 图像块在每个通道上的像素均值
        corrected_channel_means = compute_channel_means(corrected_visible_patches)

        # 计算差异度
        loss = sum([abs(corrected_mean - target_mean) for corrected_mean, target_mean in
                    zip(corrected_channel_means, target_channel_means)])
        return loss

    # 初始的gamma值（根据需要调整）
    initial_gamma_values = [0.5, 0.5, 0.5]

    # 最小化损失函数，找到最佳的gamma值
    result = minimize(loss_function, initial_gamma_values, args=(visible_patches, target_channel_means), method='Nelder-Mead')
    best_gamma_values = result.x

    return best_gamma_values


def save_patches_to_folders(patch_tuples, dataset_type, output_folder):
    """
    将图像块保存到指定的文件夹中

    参数：
    - patch_tuples：包含文件名和图像块内容的元组列表
    - dataset_type：数据集类型（目标数据集、新数据集经Gamma校正、未选择的新数据集）
    - output_folder：要保存到的文件夹路径
    """
    # 创建用于保存特定类型数据集的文件夹路径
    dataset_folder = os.path.join(output_folder, dataset_type)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # 保存每个图像块到文件夹中
    for filename, patch in patch_tuples:
        cv2.imwrite(os.path.join(dataset_folder, filename), patch)


def restore_images(selected_patches, unselected_patches, output_folder):
    """
    将裁剪、挑选、伽马校正后的图像恢复为原始大小并保存到指定文件夹

    参数：
    - selected_patches：已选择的图像块列表，每个元素为(filename, patch)元组
    - unselected_patches：未选择的图像块列表，每个元素为(filename, patch)元组
    - output_folder：保存恢复后的原始大小图像的文件夹路径

    """
    # 创建用于保存恢复后图像的文件夹路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有的原始文件名
    original_filenames = set([filename.split('_patch')[0] for filename, _ in selected_patches + unselected_patches])

    # 遍历所有的原始文件名，对每个原始文件名执行恢复操作
    for original_filename in original_filenames:
        # print('original_filename', original_filename)

        # 初始化原始大小的图像，全部为零
        restored_image = np.zeros((768, 1024, 3), dtype=np.uint8)

        # 匹配文件名的正则表达式
        pattern = re.compile(f"{original_filename}_patch_(\d+)_(\d+)\.png")

        # 将已选择和未选择的图像块放置到原始大小的图像中
        for patches in [selected_patches, unselected_patches]:
            for filename, patch in patches:
                # 如果当前图像块的文件名与当前原始文件名匹配，则将其放置到恢复后的图像中
                if filename.startswith(original_filename):
                    # 匹配文件名中的坐标信息
                    match = pattern.match(filename)
                    if match:
                        x, y = map(int, match.groups())
                        # 检查图像块是否超出图像的尺寸
                        if y + patch.shape[0] <= restored_image.shape[0] and x + patch.shape[1] <= restored_image.shape[1]:
                            restored_image[y:y+patch.shape[0], x:x+patch.shape[1]] = patch

        # 保存恢复后的图像，使用原始文件名
        cv2.imwrite(os.path.join(output_folder, f"{original_filename}_restored.png"), restored_image)


# Example usage:
dataset_folder1 = './dataset/RoadScene/'
dataset_train_folder1 = './dataset/RoadScene/train/'
dataset_folder2 = './dataset/M3FD_Fusion/'
image_size = (64, 64)  # Adjust patch size as needed
stride = 200

# Crop datasets
dataset1_visible_patches, dataset1_infrared_patches = crop_dataset(dataset_folder1, image_size, stride)
dataset1_train_visible_patches, dataset1_train_infrared_patches = crop_dataset(dataset_train_folder1, image_size,
                                                                               stride)
dataset2_visible_patches, dataset2_infrared_patches = crop_dataset(dataset_folder2, image_size, stride)

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
k = 800  # 选择top-k数量，根据需要调整
top_k_dataset2_visible_patches, top_k_dataset2_infrared_patches = top_k_patches(dataset2_visible_patches, dataset2_infrared_patches, dataset2_differences, k)

top_k_dataset2_visible_patches_channel_means = compute_channel_means(top_k_dataset2_visible_patches)
print('Top-k_patches channel means', top_k_dataset2_visible_patches_channel_means)

# 优化gamma值
best_gamma_values = optimize_gamma(top_k_dataset2_visible_patches, dataset1_channel_means)

# 应用最佳的gamma值到图像块
best_gamma_B, best_gamma_G, best_gamma_R = best_gamma_values
gamma_corrected_patches = apply_gamma_correction_to_patches(top_k_dataset2_visible_patches, best_gamma_B, best_gamma_G, best_gamma_R)
gamma_corrected_patches_channel_means = compute_channel_means(gamma_corrected_patches)
print("Gamma_corrected_patches channel means:", gamma_corrected_patches_channel_means)

# 输出最佳的gamma值
print("Best gamma values:", best_gamma_values)

# 合并数据集
merged_dataset_visible_patches = dataset1_train_visible_patches + top_k_dataset2_visible_patches
merged_dataset_infrared_patches = dataset1_train_infrared_patches + top_k_dataset2_infrared_patches

# 创建保存合并数据集的文件夹
merged_dataset_folder = './dataset/merged/'
if not os.path.exists(merged_dataset_folder):
    os.makedirs(merged_dataset_folder)

# 保存各个类型的图像块到相应的文件夹中
save_patches_to_folders(merged_dataset_visible_patches, "vis", merged_dataset_folder)
save_patches_to_folders(merged_dataset_infrared_patches, "ir", merged_dataset_folder)

