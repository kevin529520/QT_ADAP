# import os
# import numpy as np
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
#                             QVBoxLayout, QHBoxLayout, QPushButton, 
#                             QFileDialog, QMessageBox, QLabel, 
#                             QCheckBox, QColorDialog)
# from PyQt5.QtCore import Qt
# import open3d as o3d
# import sys
# import threading

# class PointCloudViewer(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#         self.pointcloud = None
#         self.vis = None
        
#     def initUI(self):
#         self.setWindowTitle("点云查看器")
#         self.setGeometry(100, 100, 400, 100)
        
#         # 创建主窗口布局
#         self.main_widget = QWidget()
#         self.setCentralWidget(self.main_widget)
#         self.layout = QVBoxLayout(self.main_widget)
        
#         # 创建控制按钮
#         self.createControls()
        
#     def createControls(self):
#         # 控制按钮布局
#         self.control_widget = QWidget()
#         self.control_layout = QHBoxLayout(self.control_widget)
        
#         # 添加按钮
#         self.load_button = QPushButton("加载点云")
#         self.color_button = QPushButton("更改颜色")
#         self.downsample_checkbox = QCheckBox("降采样")
        
#         # 连接信号
#         self.load_button.clicked.connect(self.load_pointcloud)
#         self.color_button.clicked.connect(self.change_color)
#         self.downsample_checkbox.stateChanged.connect(self.update_display)
        
#         # 添加到布局
#         self.control_layout.addWidget(self.load_button)
#         self.control_layout.addWidget(self.color_button)
#         self.control_layout.addWidget(self.downsample_checkbox)
#         self.layout.addWidget(self.control_widget)
        
#         # 添加使用说明标签
#         self.help_label = QLabel(
#             "操作说明:\n"
#             "左键点击并拖动: 旋转\n"
#             "右键点击并拖动: 平移\n"
#             "鼠标滚轮: 缩放\n"
#             "Shift + 左键点击: 改变旋转中心"
#         )
#         self.layout.addWidget(self.help_label)

#     def load_pointcloud(self):
#         file_path, _ = QFileDialog.getOpenFileName(
#             self, 
#             "打开点云文件", 
#             "", 
#             "点云文件 (*.pcd *.ply)"
#         )
#         if file_path:
#             try:
#                 # 读取点云
#                 pcd = o3d.io.read_point_cloud(file_path)
                
#                 # 如果选中了降采样
#                 if self.downsample_checkbox.isChecked():
#                     # 计算合适的体素大小
#                     bbox = pcd.get_axis_aligned_bounding_box()
#                     bbox_extent = bbox.get_extent()
#                     voxel_size = np.min(bbox_extent) / 50
#                     pcd = pcd.voxel_down_sample(voxel_size)
                
#                 # 显示点云信息
#                 points_num = len(pcd.points)
#                 self.statusBar().showMessage(f"已加载点云，共 {points_num} 个点")
                
#                 # 在新线程中显示点云
#                 self.pointcloud = pcd
#                 threading.Thread(target=self.show_pointcloud).start()
                
#             except Exception as e:
#                 QMessageBox.critical(self, "错误", f"无法加载点云文件: {str(e)}")
    
#     def show_pointcloud(self):
#         if self.vis is not None:
#             self.vis.destroy_window()
            
#         self.vis = o3d.visualization.Visualizer()
#         self.vis.create_window("点云查看器")
        
#         # 设置渲染选项
#         opt = self.vis.get_render_option()
#         opt.point_size = 2.0
#         opt.background_color = np.asarray([0.1, 0.1, 0.1])
        
#         # 添加点云
#         self.vis.add_geometry(self.pointcloud)
        
#         # 设置默认视角
#         self.vis.reset_view_point(True)
        
#         # 运行可视化器
#         self.vis.run()
#         self.vis.destroy_window()
    
#     def update_display(self):
#         if self.pointcloud is not None:
#             self.load_pointcloud()  # 重新加载点云以应用新的设置
    
#     def change_color(self):
#         if self.pointcloud is not None:
#             color = QColorDialog.getColor()
#             if color.isValid():
#                 self.pointcloud.paint_uniform_color([
#                     color.redF(), 
#                     color.greenF(), 
#                     color.blueF()
#                 ])
#                 # 更新显示
#                 if self.vis is not None:
#                     self.vis.update_geometry(self.pointcloud)
#                     self.vis.poll_events()
#                     self.vis.update_renderer()
    
#     def closeEvent(self, event):
#         if self.vis is not None:
#             self.vis.destroy_window()
#         event.accept()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     viewer = PointCloudViewer()
#     viewer.show()
#     sys.exit(app.exec_())

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import open3d as o3d
import numpy as np
import threading
import os

class PointCloudViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.init_ui()
        self.pointcloud = None
        self.vis = None
        
    def init_ui(self):
        # 设置窗口
        self.root.title("点云查看器")
        self.root.geometry("400x200")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 加载点云按钮
        self.load_button = ttk.Button(
            button_frame, 
            text="加载点云", 
            command=self.load_pointcloud
        )
        self.load_button.grid(row=0, column=0, padx=5)
        
        # 更改颜色按钮
        self.color_button = ttk.Button(
            button_frame, 
            text="更改颜色", 
            command=self.change_color
        )
        self.color_button.grid(row=0, column=1, padx=5)
        
        # 降采样复选框
        self.downsample_var = tk.BooleanVar()
        self.downsample_check = ttk.Checkbutton(
            button_frame,
            text="降采样",
            variable=self.downsample_var,
            command=self.update_display
        )
        self.downsample_check.grid(row=0, column=2, padx=5)
        
        # 帮助信息
        help_text = (
            "操作说明:\n"
            "左键点击并拖动: 旋转\n"
            "右键点击并拖动: 平移\n"
            "鼠标滚轮: 缩放\n"
            "Shift + 左键点击: 改变旋转中心"
        )
        help_label = ttk.Label(main_frame, text=help_text, justify=tk.LEFT)
        help_label.grid(row=1, column=0, pady=10)
        
        # 状态栏
        self.status_var = tk.StringVar()
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=2, column=0, pady=5)
        
        # 设置列和行的权重
        main_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure((0,1,2), weight=1)
        
    def load_pointcloud(self):
        file_path = filedialog.askopenfilename(
            title="打开点云文件",
            filetypes=[("点云文件", "*.pcd *.ply")]
        )
        
        if file_path:
            try:
                # 读取点云
                pcd = o3d.io.read_point_cloud(file_path)
                
                # 如果选中了降采样
                if self.downsample_var.get():
                    # 计算合适的体素大小
                    bbox = pcd.get_axis_aligned_bounding_box()
                    bbox_extent = bbox.get_extent()
                    voxel_size = np.min(bbox_extent) / 50
                    pcd = pcd.voxel_down_sample(voxel_size)
                
                # 显示点云信息
                points_num = len(pcd.points)
                self.status_var.set(f"已加载点云，共 {points_num} 个点")
                
                # 在新线程中显示点云
                self.pointcloud = pcd
                threading.Thread(target=self.show_pointcloud).start()
                
            except Exception as e:
                messagebox.showerror("错误", f"无法加载点云文件: {str(e)}")
    
    def show_pointcloud(self):
        if self.vis is not None:
            self.vis.destroy_window()
            
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("点云查看器")
        
        # 设置渲染选项
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        
        # 添加点云
        self.vis.add_geometry(self.pointcloud)
        
        # 设置默认视角
        self.vis.reset_view_point(True)
        
        # 运行可视化器
        self.vis.run()
        self.vis.destroy_window()
    
    def update_display(self):
        if self.pointcloud is not None:
            self.load_pointcloud()  # 重新加载点云以应用新的设置
    
    def change_color(self):
        if self.pointcloud is not None:
            # 创建颜色选择器窗口
            color_picker = tk.Toplevel(self.root)
            color_picker.title("选择颜色")
            
            def update_color(color):
                try:
                    # 将颜色字符串转换为RGB值
                    r = int(color[1:3], 16) / 255.0
                    g = int(color[3:5], 16) / 255.0
                    b = int(color[5:7], 16) / 255.0
                    
                    self.pointcloud.paint_uniform_color([r, g, b])
                    
                    # 更新显示
                    if self.vis is not None:
                        self.vis.update_geometry(self.pointcloud)
                        self.vis.poll_events()
                        self.vis.update_renderer()
                    
                    color_picker.destroy()
                except Exception as e:
                    messagebox.showerror("错误", f"无法更改颜色: {str(e)}")
            
            # 使用tkinter的颜色选择器
            from tkinter.colorchooser import askcolor
            color = askcolor(title="选择点云颜色")
            if color[1]:  # 如果用户选择了颜色
                update_color(color[1])
            color_picker.destroy()
    
    def run(self):
        self.root.mainloop()
        
    def on_closing(self):
        if self.vis is not None:
            self.vis.destroy_window()
        self.root.destroy()

if __name__ == "__main__":
    viewer = PointCloudViewer()
    viewer.root.protocol("WM_DELETE_WINDOW", viewer.on_closing)
    viewer.run()