import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
import cv2
from modules.train_resNet import ResNet  # 从训练代码中导入ResNet模型
from weldPosPredict import weldPosPredict


# 定义预测函数
def beadProfilePredict(image, weld_pos, model_path, temp_output_path, draw_image_path):
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
    cv2.circle(draw_image, tuple(origin_weld_pos), 5, (0, 0, 255), -1)  # 绘制焊接位置（红色圆点）

    # 保存临时结果图片
    # temp_output_path = temp_output_path
    # draw_image_path = draw_image_path
    cv2.imwrite(temp_output_path, output_image)
    cv2.imwrite(draw_image_path, draw_image)

    return temp_output_path, draw_image_path, circle_center, circle_radius