import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
import cv2
from train_resNet import ResNet  # 从训练代码中导入ResNet模型

# 定义预测函数
def predict(image, weld_pos, model_path):
    """预测函数"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载并预处理输入图片
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
    temp_output_path = "./temp_output.png"
    draw_image_path = "./draw_image.png"
    cv2.imwrite(temp_output_path, output_image)
    cv2.imwrite(draw_image_path, draw_image)

    return temp_output_path, draw_image_path, circle_center, circle_radius

# GUI 应用程序
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("焊接点预测工具")

        # 初始化变量
        self.image_path = None  
        self.image = None   # 保存当前图片对象
        self.original_image_path = None  # 保存原始图片路径
        self.weld_pos = [0, 0]
        self.photo = None  # 用于保存当前显示的图片对象
        self.circle_center = None  # 拟合圆的圆心
        self.circle_radius = None  # 拟合圆的半径
        self.last_mouse_pos = None  # 上一次鼠标位置（用于优化性能）

        # 创建左侧图片显示区域
        self.image_label = tk.Label(root)
        self.image_label.grid(row=0, column=0, padx=10, pady=10)

        # 绑定鼠标移动事件
        self.image_label.bind("<Motion>", self.on_mouse_move)

        # 绑定鼠标点击事件
        self.image_label.bind("<Button-1>", self.on_mouse_click)

        # 创建右侧输入区域
        input_frame = tk.Frame(root)
        input_frame.grid(row=0, column=1, padx=10, pady=10)

        # 加载图片按钮
        tk.Button(input_frame, text="加载图片", command=self.load_image).grid(row=0, column=0, columnspan=2, pady=10)

        # 重置按钮
        tk.Button(input_frame, text="重置", command=self.reset_image).grid(row=1, column=0, columnspan=2, pady=10)

        # 创建文本显示区域
        self.info_label = tk.Label(input_frame, text="焊接位置: \n拟合圆心: \n拟合半径: ", justify=tk.LEFT)
        self.info_label.grid(row=2, column=0, columnspan=2, pady=10)

    def load_image(self):
        """加载图片并显示"""
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if self.image_path:
            self.original_image_path = self.image_path  # 保存原始图片路径
            self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图加载
            self.show_image(self.image_path)
            self.update_info()  # 清空显示信息

    def show_image(self, image_path):
        """显示图片"""
        image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def on_mouse_move(self, event):
        """鼠标移动事件"""
        if not self.image_path:
            return

        # 获取鼠标当前位置
        x, y = event.x, event.y

        # 优化性能：避免频繁预测
        if self.last_mouse_pos and abs(x - self.last_mouse_pos[0]) < 5 and abs(y - self.last_mouse_pos[1]) < 5:
            return
        self.last_mouse_pos = (x, y)

        # 更新焊接位置
        self.weld_pos = [x, y]

        # 调用预测函数
        self.run_prediction()

    def on_mouse_click(self, event):
        """鼠标点击事件"""

        if not self.image_path:
            return

        self.image = cv2.imread('./temp_output.png', cv2.IMREAD_GRAYSCALE)  # 以灰度图加载
        # print('self.image_path:', self.image_path)
        self.update_info()

    def run_prediction(self):
        """运行预测并更新图片"""
        if not self.image_path:
            return

        # 调用预测函数
        model_path = "weight_0301_mini.pth"  # 模型路径
        
        output_image_path, draw_image_path, self.circle_center, self.circle_radius = predict(
            self.image, self.weld_pos, model_path
        )

        # 更新图片显示
        self.show_image(draw_image_path)

        # 更新拟合圆信息
        self.update_info()

    def update_info(self):
        """更新文本显示区域的内容"""
        info_text = f"焊接位置: ({self.weld_pos[0]}, {self.weld_pos[1]})\n"
        if self.circle_center and self.circle_radius:
            info_text += f"拟合圆心: ({self.circle_center[0]}, {self.circle_center[1]})\n"
            info_text += f"拟合半径: {self.circle_radius}"
        else:
            info_text += "拟合圆心: \n拟合半径: "
        self.info_label.config(text=info_text)

    def reset_image(self):
        """重置图片显示"""
        if self.original_image_path:
            self.show_image(self.original_image_path)
            self.image_path = self.original_image_path  # 重置当前图片路径为原始图片
            self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.circle_center = None
            self.circle_radius = None
            self.update_info()  # 清空显示信息

# 运行应用程序
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()