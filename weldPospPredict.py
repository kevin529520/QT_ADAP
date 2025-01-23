import torch
import numpy as np
import cv2
import os
from torchvision import transforms
from modules.PoseNet import PNet  # 从训练代码中导入PNet模型

# 定义预测函数
def predict(image_path, model_path, output_image_path):
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PNet(model_path).to(device)  # 初始化模型
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载训练好的权重
    model.eval()  # 设置为评估模式

    # 加载并预处理输入图片
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图加载
    image = image / 255.0  # 归一化
    image = cv2.resize(image, (320, 320))  # 调整大小为 320x320
    image_tensor = torch.Tensor(image).unsqueeze(0).unsqueeze(0).to(device)  # 转换为张量并添加批次维度

    # 进行预测
    with torch.no_grad():
        output = model(image_tensor, None, None)
        output = output.cpu().numpy().flatten()  # 将输出转换为numpy数组

    # 解析预测结果
    weld_position = (int(output[0]), int(output[1]))  # 焊接位置坐标 (x, y)
    circle_center = (int(output[2]), int(output[3]))  # 圆心坐标 (x, y)
    circle_radius = int(output[4])  # 圆半径

    # 在图片上绘制焊接位置
    output_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 重新加载原始图片
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图以便绘制彩色标记
    cv2.circle(output_image, weld_position, 5, (0, 0, 255), -1)  # 绘制焊接位置（红色圆点）

    # 保存结果图片
    cv2.imwrite(output_image_path, output_image)

    # 返回预测结果
    return weld_position, circle_center, circle_radius


# 主程序
if __name__ == "__main__":
    # 输入参数
    img_folder = './Resnet_var'
    

    img_name = "00000112.jpg"  # 输入图片路径
    image_path = os.path.join(img_folder, img_name)
    output_image_path = "./Resnet_var/group2/0112_weldPos.png"  # 输出图片路径

    img_folder = r'D:\vscodeproject\alexnet\VAE_var\group2'
    output_folder = './Resnet_var/group2'
    img_name = '1512_resnetPredict.png'
    image_path = os.path.join(img_folder, img_name)
    output_image_name = img_name.split('_')[0] + '_weldPos.png'
    output_image_path = os.path.join(output_folder, output_image_name)

    # image_path = "./Resnet_var/00000112.jpg"  # 输入图片路径
    # image_path = "./Resnet_var/0312_resnetPredict.jpg"  # 输入图片路径
    # image_path = "./Resnet_var/0512_resnetPredict.jpg"  # 输入图片路径
    # image_path = "./Resnet_var/0712_resnetPredict.jpg"  # 输入图片路径
    # image_path = "./Resnet_var/0912_resnetPredict.jpg"  # 输入图片路径
    # image_path = "./Resnet_var/1112_resnetPredict.jpg"  # 输入图片路径
    # image_path = "./Resnet_var/1312_resnetPredict.jpg"  # 输入图片路径
    # image_path = "./Resnet_var/1512_resnetPredict.jpg"  # 输入图片路径
    model_path = "./weights/lr_0.0001_epo2000/pose_weight.pth"  # 训练好的模型权重路径
    

    # 进行预测
    weld_pos, circle_center, circle_radius = predict(image_path, model_path, output_image_path)

    # 打印预测结果
    print(f"Predicted Weld Position: {weld_pos}")
    print(f"Predicted Circle Center: {circle_center}")
    print(f"Predicted Circle Radius: {circle_radius}")
    print(f"Output image saved to {output_image_path}")