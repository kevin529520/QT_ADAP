import tkinter as tk
from tkinter import filedialog, messagebox
import open3d as o3d
import numpy as np
from PIL import Image, ImageTk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Viewer")
        self.root.geometry("1000x1000") 

        # 创建左侧点云显示区域
        self.pointcloud_frame = tk.Frame(root, width=500, height=500)
        self.pointcloud_frame.grid(row=0, column=0, padx=10, pady=10)
        self.pointcloud_frame.grid_propagate(False)  # 防止框架根据其内容自动调整大小

        self.blank_image = Image.new("L", (400, 400), 255)
        self.blank_photo = ImageTk.PhotoImage(self.blank_image)

        # 点云显示区域
        self.pointcloud_label = tk.Label(self.pointcloud_frame, image = self.blank_photo , width=400, height=400)
        self.pointcloud_label.pack(expand=True, fill=tk.BOTH)

        # 加载点云按钮
        self.load_button = tk.Button(root, text="加载点云", command=self.load_pointcloud)
        self.load_button.grid(row=1, column=0, padx=10, pady=10)

    def load_pointcloud(self):
        """加载点云数据"""
        file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.pcd *.ply")])
        if file_path:
            try:
                # 读取点云文件
                self.pointcloud = o3d.io.read_point_cloud(file_path)
                
                # 创建可视化器
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(self.pointcloud)
                
                # 设置相机视角
                ctr = vis.get_view_control()
                ctr.set_zoom(0.8)
                
                # 渲染点云到图像
                vis.poll_events()
                vis.update_renderer()
                image = vis.capture_screen_float_buffer(do_render=True)
                vis.destroy_window()
                
                # 转换为PIL图像
                image = (np.asarray(image) * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
                
                # 显示在Tkinter中
                # pointcloud_photo = ImageTk.PhotoImage(pil_image)
                # pointcloud_photo = ImageTk.PhotoImage()
                
                self.blank_image = Image.new("L", (600, 600), 255)
                # self.blank_image.resize((800, 700), Image.ANTIALIAS)
                self.blank_photo = ImageTk.PhotoImage(self.blank_image)
                # self.pointcloud_label.config(image=pointcloud_photo, width=480, height=450)
                self.pointcloud_label.config(image=self.blank_photo, width=600, height=600)
                self.pointcloud_label.image = self.blank_photo
                
                # 更新状态
                # self.pointcloud_label.config(text="")

                # self.root.geometry("1500x650") 
                self.root.geometry("1500x1500") 
                # self.pointcloud_frame = tk.Frame(root, width=480, height=550)
                self.pointcloud_frame = tk.Frame(root, width=750, height= 750)
                self.load_button.grid(row=1, column=0, padx=15, pady=15)
            except Exception as e:
                messagebox.showerror("错误", f"无法加载点云文件: {str(e)}")
                self.pointcloud_label.config(text="点云加载失败")

# 创建主窗口
root = tk.Tk()
app = App(root)

# 运行主循环
root.mainloop()