import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
import cv2
from modules.train_resNet import ResNet  # 从训练代码中导入ResNet模型
from beadProfilePredict import beadProfilePredict
from weldPosPredict import weldPosPredict
from point2image import PointCloudProcessor
import open3d as o3d

# GUI 应用程序
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("焊接点预测工具")
        self.root.geometry("1500x650")  # Set initial window size (wider to accommodate buttons)
        
        # Create blank grayscale image (white background for binary images)
        self.blank_image = Image.new("L", (480, 450), 255)
        self.blank_photo = ImageTk.PhotoImage(self.blank_image)

        # 初始化变量
        self.image_path = None  # 保存焊道图片（Bx）路径
        self.temp_image_path = './images/temp_output.png'  # 保存临时图片路径
        self.draw_image_path = './images/draw_image.png'  # 保存绘制图片路径
        self.weldPos_image_path = './images/temp_weld_pos.png'  # 保存焊接位置图片路径
        self.image = None   # 保存当前图片对象
        self.original_image_path = None  # 保存原始图片路径
        self.weld_pos = [0, 0]
        self.photo = None  # 用于保存当前显示的图片对象
        self.circle_center = None  # 拟合圆的圆心
        self.circle_radius = None  # 拟合圆的半径
        self.last_mouse_pos = None  # 上一次鼠标位置（用于优化性能）
        self.pointcloud = None  # 保存当前点云对象
        self.workpiece = None  # 保存工件对象
        self.x_crosssection = None  # 保存焊道横截面位置 
        self.projected_image_path = None  # 保存点云转换的图像
        self.processed_image_path = None  # 保存处理后的图像

        # 创建左侧点云显示区域
        self.pointcloud_frame = tk.Frame(root, width=480, height=550)
        self.pointcloud_frame.grid(row=0, column=0, padx=10, pady=10)
        self.pointcloud_frame.grid_propagate(False)  # Prevent frame from resizing
        
        # 点云显示区域
        # self.pointcloud_label = tk.Label(self.pointcloud_frame, width=480, height=450)
        self.pointcloud_label = tk.Label(self.pointcloud_frame, image = self.blank_photo, width=480, height=450)
        self.pointcloud_label.pack()
        self.pointcloud_label.image = self.blank_photo  # Keep reference
        # self.pointcloud_label.pack(expand=True, fill=tk.BOTH)
        
        # 点云下方文字
        self.pointcloud_caption = tk.Label(self.pointcloud_frame, text="焊缝点云", font=("Arial", 12))
        self.pointcloud_caption.pack(pady=5)
        
        # 创建右侧图片显示区域
        self.image_frame = tk.Frame(root)
        self.image_frame.grid(row=0, column=1, padx=10, pady=10)
        
        # 图片显示区域
        self.image_label = tk.Label(self.image_frame, image=self.blank_photo)
        self.image_label.pack()
        self.image_label.image = self.blank_photo  # Keep reference
        
        # 图片下方文字
        self.image_caption = tk.Label(self.image_frame, text="焊道截面", font=("Arial", 12))
        self.image_caption.pack(pady=5)

        # 添加滑动条
        self.slider = tk.Scale(root, from_=0, to=30, orient=tk.HORIZONTAL, label="X Cross-section", command=self.update_crosssection)
        self.slider.grid(row=1, column=1, padx=10, pady=10)

        # 绑定鼠标移动事件
        self.image_label.bind("<Motion>", self.on_mouse_move)

        # 绑定鼠标点击事件
        self.image_label.bind("<Button-1>", self.on_mouse_click)

        # 创建右侧输入区域
        input_frame = tk.Frame(root)
        input_frame.grid(row=0, column=2, padx=10, pady=10)
        
        # 添加点云加载按钮
        tk.Button(input_frame, text="加载点云", command=self.load_pointcloud).grid(row=0, column=0, columnspan=2, pady=10)

        # 加载图片按钮
        tk.Button(input_frame, text="加载图片", command=self.load_image).grid(row=1, column=0, columnspan=2, pady=10)

        # 重置按钮
        tk.Button(input_frame, text="重置", command=self.reset_image).grid(row=2, column=0, columnspan=2, pady=10)

        # 焊接位置预测按钮
        tk.Button(input_frame, text="焊接位置预测", command=self.predict_weld_position).grid(row=3, column=0, columnspan=2, pady=10)

        # 确认预测按钮
        tk.Button(input_frame, text="确认预测", command=self.confirm_prediction).grid(row=4, column=0, columnspan=2, pady=10)

        # 创建文本显示区域
        self.info_label = tk.Label(input_frame, text="焊接位置: \n拟合圆心: \n拟合半径: ", justify=tk.LEFT)
        self.info_label.grid(row=5, column=0, columnspan=2, pady=10)

    def load_pointcloud(self):
        """加载点云数据"""
        file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.pcd *.ply")])
        if file_path:
            try:
                # 读取点云文件
                self.pointcloud = o3d.io.read_point_cloud(file_path)
                self.workpiece = file_path.split("/")[-1]
                
                # 创建可视化器
                vis = o3d.visualization.Visualizer()
                # vis.create_window(width=480, height=550, visible=True)
                vis.create_window(width=480, height=550, visible=False)
                vis.add_geometry(self.pointcloud)
                
                # 设置相机视角
                ctr = vis.get_view_control()
                # ctr.set_zoom(0.8)
                
                # 渲染点云到图像
                vis.poll_events()
                vis.update_renderer()
                image = vis.capture_screen_float_buffer(do_render=False)
                vis.destroy_window()
                
                # # 转换为PIL图像
                image = (np.asarray(image) * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
                pil_image = pil_image.resize((480, 450))
                
                # # 显示在Tkinter中
                pointcloud_photo = ImageTk.PhotoImage(pil_image)
                self.pointcloud_label.config(image=pointcloud_photo, width=480, height=450)
                self.pointcloud_label.image = pointcloud_photo
                # self.pointcloud_label.image = self.blank_photo  # Keep reference
                # self.pointcloud_label.config(width=480, height=450)
                
                # 更新状态
                self.pointcloud_label.config(text="")
            except Exception as e:
                messagebox.showerror("错误", f"无法加载点云文件: {str(e)}")
                self.pointcloud_label.config(text="点云加载失败")

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
        if self.image is None:
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
        if not self.image.any():
            return

        # 更新图像为最新的预测结果
        self.image = cv2.imread(self.temp_image_path, cv2.IMREAD_GRAYSCALE)
        self.image_path = self.temp_image_path  # 更新图像路径
        self.update_info()

    def run_prediction(self):
        """运行预测并更新图片"""
        # print("self.image:", self.image)
        if not self.image.any():
            return

        # 调用预测函数
        model_path = "./weights/weight_0301_mini.pth"  # 模型路径
        output_image_path, draw_image_path, self.circle_center, self.circle_radius = beadProfilePredict(
            self.image, self.weld_pos, model_path, self.temp_image_path, self.draw_image_path
        )

        # 更新图片显示
        self.show_image(draw_image_path)

        # 更新拟合圆信息
        self.update_info()

    def predict_weld_position(self):
        """预测焊接位置并更新图片和信息"""
        if not self.image_path:
            return

        # 调用焊接位置预测函数
        model_path = "./weights/pose_weight.pth"  # 焊接位置模型路径
        weld_pos, circle_center, circle_radius = weldPosPredict(self.image, model_path, self.weldPos_image_path)

        # 更新焊接位置
        self.weld_pos = weld_pos

        # 更新图片显示
        self.show_image(self.weldPos_image_path)

        self.update_info()

    def confirm_prediction(self):
        """确认预测并更新图像"""
        if not self.image_path:
            return

        # 调用预测函数
        self.run_prediction()

        self.image = cv2.imread(self.temp_image_path, cv2.IMREAD_GRAYSCALE)
        # 更新拟合圆信息

    def update_crosssection(self, value):
        """更新x_crosssection数值并调用point2image函数"""
        self.x_crosssection = int(value)
        self.point2image()

    def point2image(self):
        """将点云转换为图像"""
        if not self.pointcloud:
            return

        # 创建点云处理器
        processor = PointCloudProcessor(self.workpiece, self.x_crosssection)
        print(f"Processing point cloud of {self.workpiece} at x={self.x_crosssection}...")

        # 创建输出目录
        output_dir = os.path.join('images', self.workpiece.split('.')[0])
        os.makedirs(output_dir, exist_ok=True)

        # 设置输出路径
        self.projected_image_path = os.path.join(output_dir, 'projected_image.png')
        self.processed_image_path = os.path.join(output_dir, 'processed_image.png')

        # 将点云转换为图像
        image = processor.project_to_yz_plane(self.x_crosssection)
        cv2.imwrite(self.projected_image_path, image)

        processor.fill_point_cloud_section(self.projected_image_path, self.processed_image_path)

        # 显示处理后的图像
        try:
            image = cv2.imread(self.processed_image_path, cv2.IMREAD_GRAYSCALE)
            self.image = image
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("错误", f"显示图像失败: {str(e)}")

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
        else:
            self.show_image(self.processed_image_path)
            self.image = cv2.imread(self.processed_image_path, cv2.IMREAD_GRAYSCALE)
            self.circle_center = None
            self.circle_radius = None
            self.update_info()  # 清空显示信息

# 运行应用程序
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
