import os
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
import cv2
from modules.train_resNet import ResNet  # 从训练代码中导入ResNet模型
from weldPosPredict import weldPosPredict

def calculate_surface_smoothness(image):
    # 获取图像尺寸
    height, width = image.shape
    
    # 存储每列第一个非黑色像素的y坐标
    surface_heights = []
    x_positions = []
    
    # 对每列进行遍历
    for x in range(width):
        # 获取当前列
        column = image[:, x]
        # 找到第一个非黑色像素的索引
        non_black_indices = np.where(column > 0)[0]
        if len(non_black_indices) > 0:
            surface_heights.append(non_black_indices[0])
            x_positions.append(x)
    
    # 计算高度的统计特征
    heights_array = np.array(surface_heights)
    # mean_height = np.mean(heights_array)
    # height_variance = np.var(heights_array)
    height_std = np.std(heights_array)
    
    # 可视化结果
    # plt.figure(figsize=(12, 6))
    
    # # 左图：显示原始图像和检测到的表面线
    # plt.subplot(121)
    # plt.imshow(image, cmap='gray')
    # plt.plot(x_positions, surface_heights, 'r-', label='Surface Line')
    # plt.axhline(y=mean_height, color='g', linestyle='--', label='Mean Height')
    # plt.legend()
    # plt.title('Surface Detection')
    
    # # 右图：显示高度分布
    # plt.subplot(122)
    # plt.plot(x_positions, surface_heights)
    # plt.axhline(y=mean_height, color='r', linestyle='--', label='Mean Height')
    # plt.fill_between(x_positions, 
    #                  mean_height - height_std, 
    #                  mean_height + height_std, 
    #                  alpha=0.2, 
    #                  color='r', 
    #                  label='Standard Deviation')
    # plt.legend()
    # plt.title(f'Height Distribution\nVariance: {height_variance:.2f}')
    # plt.xlabel('X Position')
    # plt.ylabel('Height')
    
    # plt.tight_layout()
    # plt.show()
    
    return height_std, x_positions, surface_heights
    # return {
    #     'mean_height': mean_height,
    #     'height_variance': height_variance,
    #     'height_std': height_std
    # }


# 定义预测函数
def beadProfilePredict(image, weld_pos, model_path, temp_output_path, draw_image_path, processed_image_path, original_image_path):
    """预测函数"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载并预处理输入图片
    original_image = image.copy()
    image = image / 255.0
    image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
    image_tensor = torch.Tensor(image).unsqueeze(0).unsqueeze(0).to(device)

    # 预处理焊接点坐标
    origin_weld_pos = weld_pos
    weld_pos = np.array(weld_pos) * 0.5
    weld_pos_tensor = torch.Tensor(weld_pos).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        output = model(image_tensor, weld_pos_tensor)
        output = output.cpu().numpy().flatten()

    # 解析预测结果
    circle_center = (round(output[0] * 2), round(output[1] * 2))
    circle_radius = round(output[2] * 2)

    # 在图片上绘制圆
    output_image = original_image.copy()
    mask = np.zeros_like(output_image)
    cv2.circle(mask, circle_center, circle_radius, (255, 255, 255), -1)
    output_image[mask == 255] = 0

    # 绘制焊接位置
    mask2 = original_image - output_image
    draw_image = original_image.copy()
    draw_image[mask2 == 255] = 100

    processed_image = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
    if processed_image is not None:
        image_2 = processed_image - draw_image
    else:   
        init_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        image_2 = init_image - draw_image
    # 计算平整度
    height_std, x_positions, surface_heights = calculate_surface_smoothness(image_2)

    # 定义一连串点的位置
    points = [(x, y) for x, y in zip(x_positions, surface_heights)]  # 示例坐标

    # 定义点的颜色和半径
    color = (0, 255, 0)  # BGR 颜色 (红色)
    radius = 2  # 圆点半径
    thickness = -1  # 填充圆点

    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_GRAY2BGR)

    # 在图像上绘制一连串的点
    for point in points:
        cv2.circle(draw_image, point, radius, color, thickness)

    # cv2.circle(draw_image, tuple(origin_weld_pos), 5, (0, 0, 255), -1)  # 绘制焊接位置（红色圆点）

    cv2.circle(draw_image, tuple(origin_weld_pos), 10, (0, 0, 255), 2)  # 绘制圆（红色边框）
    cv2.line(draw_image, (origin_weld_pos[0] - 7, origin_weld_pos[1] - 7), (origin_weld_pos[0] + 7, origin_weld_pos[1] + 7), (0, 0, 255), 2)  # 绘制 'x' 的一条线
    cv2.line(draw_image, (origin_weld_pos[0] - 7, origin_weld_pos[1] + 7), (origin_weld_pos[0] + 7, origin_weld_pos[1] - 7), (0, 0, 255), 2)  # 绘制 'x' 的另一条线


    # 保存临时结果图片
    # temp_output_path = temp_output_path
    # draw_image_path = draw_image_path
    cv2.imwrite(temp_output_path, output_image)
    cv2.imwrite(draw_image_path, draw_image)

    # height_std = 0  
    return temp_output_path, draw_image_path, circle_center, circle_radius, height_std