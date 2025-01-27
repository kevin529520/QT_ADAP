import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

class PointCloudProcessor:
    def __init__(self, workpiece, x_crosssection = 8, y1_crop=- 22.1, y2_crop=2.443, resolution=23 / 450, x_threshold=0.5, z_range=23):
        self.workpiece = workpiece
        self.x_crosssection = x_crosssection
        self.y1_crop = y1_crop
        self.y2_crop = y2_crop
        self.resolution = resolution
        self.x_threshold = x_threshold
        self.z_range = z_range
        self.pcd = o3d.io.read_point_cloud('./pointcloud/' + workpiece)
        self.transformed_pcd = o3d.io.read_point_cloud('./pointcloud/transformedPcd/' + workpiece.split('.')[0] + '/transformed_pcd.pcd')
        self.rotation_matrix = None
        self.translation_vector = None

    def project_to_yz_plane(self, x_crosssection):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        # o3d.visualization.draw_geometries([self.transformed_pcd, frame], window_name="Transformed Point Cloud")
        points = np.asarray(self.transformed_pcd.points)
        # print("Points dimensions:", points.shape)
        
        mask = (np.abs(points[:, 0]) > x_crosssection) & (np.abs(points[:, 0]) < (x_crosssection + self.x_threshold))
        yz_points = points[mask][:, 1:]
        
        if len(yz_points) == 0:
            print("没有找到在指定范围内的点！")
            return None
        
        y_min, z_min = np.min(yz_points, axis=0)
        y_max, z_max = np.max(yz_points, axis=0)
        print(f"z范围: {z_min:.2f} to {z_max:.2f}, y范围: {y_min:.2f} to {y_max:.2f}")
        
        # height = int(self.z_range / self.resolution) + 1
        # width = int((self.y2_crop - self.y1_crop) / self.resolution) + 1
        height = int(self.z_range / self.resolution) 
        width = int((self.y2_crop - self.y1_crop) / self.resolution) 
        
        image = np.zeros((height, width), dtype=np.uint8)
        
        for point in yz_points:
            y, z = point
            pixel_x = int((y - self.y1_crop) / self.resolution)
            pixel_y = int(z / self.resolution)
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                image[height - 1 - pixel_y, pixel_x] = 255

        return image

    def transform_point_cloud(self):
        points = np.asarray(self.pcd.points)
        transformed_points = np.dot(points, self.rotation_matrix) - np.dot(self.translation_vector, self.rotation_matrix)
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        self.transformed_pcd = transformed_pcd

    def process_point_cloud(self):
        try:
            y1_crop = 440
            y2_crop = 540 
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(40, y1_crop, -30), max_bound=(70, y2_crop, 50))

            # Crop the point cloud using the bounding box
            self.pcd = self.pcd.crop(bbox)

            plane_model_1, inliers_1 = self.pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
            inlier_cloud_1 = self.pcd.select_by_index(inliers_1)
            outlier_cloud_1 = self.pcd.select_by_index(inliers_1, invert=True)

            plane_model_2, inliers_2 = outlier_cloud_1.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
            inlier_cloud_2 = outlier_cloud_1.select_by_index(inliers_2)
            outlier_cloud_2 = outlier_cloud_1.select_by_index(inliers_2, invert=True)

            normal_1 = np.array(plane_model_1[:3])
            normal_2 = np.array(plane_model_2[:3])

            normal_1 = normal_1 / np.linalg.norm(normal_1)
            normal_2 = normal_2 / np.linalg.norm(normal_2)
            print("Normal 1:", normal_1)
            print("Normal 2:", normal_2)

            z_axis = -normal_1
            y_axis = normal_2


            # y_axis =  [ 0.99766364, -0.00293591, -0.0682543 ]
            # z_axis = [0.00127739, 0.00867154, 0.99996159]

            y_axis = [ 0.99794045, -0.00203514, -0.06411492]
            z_axis = [1.65962794e-02, 2.07392313e-05, 9.99862272e-01]

            # x_axis = np.cross(normal_1, normal_2)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)

            self.rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).T

            points = np.asarray(self.pcd.points)
            max_z_point = points[np.argmax(points[:, 2])]
            min_z_point = [ 59.00999832, 194.30000305, -20.86775662]
            min_z_point = [ 60.36000061, 447.,         -21.76]
            self.translation_vector = min_z_point

            frames = []
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
            frames.append(camera_frame)
            workpiece_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=max_z_point)
            workpiece_frame.rotate(self.rotation_matrix, center=max_z_point)
            frames.append(workpiece_frame)

            o3d.visualization.draw_geometries([inlier_cloud_1, outlier_cloud_1] + [frames[0]], window_name="Plane 1")

            self.transform_point_cloud()

            o3d.visualization.draw_geometries([self.pcd, self.transformed_pcd, camera_frame], window_name="Transformed Point Cloud")

            if not os.path.exists("./pointcloud/transformedPcd"):
                os.makedirs("./pointcloud/transformedPcd")
            if not os.path.exists("./pointcloud/transformedPcd/" + self.workpiece.split('.')[0]):
                os.makedirs("./pointcloud/transformedPcd/" + self.workpiece.split('.')[0])
            o3d.io.write_point_cloud("./pointcloud/transformedPcd/" + self.workpiece.split('.')[0] + "/transformed_pcd.pcd", self.transformed_pcd)

            transform_params = {
                'rotation_matrix': self.rotation_matrix,
                'translation_vector': self.translation_vector,
                'normal_1': normal_1,
                'normal_2': normal_2,
                'x_axis': x_axis,
                'y_axis': y_axis,
                'z_axis': z_axis
            }

            np.save("./pointcloud/transformedPcd/" + self.workpiece.split('.')[0] + "/transform_params.npy", transform_params)
            
            with open("./pointcloud/transformedPcd/" + self.workpiece.split('.')[0] + "/transform_params.txt", 'w') as f:
                for key, value in transform_params.items():
                    f.write(f"{key}:\n{value}\n\n")

            transformed_pcd = o3d.io.read_point_cloud("./pointcloud/transformedPcd/" + self.workpiece.split('.')[0] + "/transformed_pcd.pcd")
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-15, self.y1_crop, -50), max_bound=(100, self.y2_crop, 50))
            # cropped_pcd = transformed_pcd.crop(bbox)
            cropped_pcd = transformed_pcd

            workpiece_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
            o3d.visualization.draw_geometries([cropped_pcd] + [workpiece_frame], window_name="Cropped Point Cloud")

            cl, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
            denoised_pcd = cropped_pcd.select_by_index(ind)
            o3d.visualization.draw_geometries([denoised_pcd], window_name="Denoised Point Cloud")

            # self.transformed_pcd = denoised_pcd

            

            if not os.path.exists("./images/" + self.workpiece.split('.')[0]):
                os.makedirs("./images/" + self.workpiece.split('.')[0])

            x_crosssection = self.x_crosssection
            img = self.project_to_yz_plane(x_crosssection)
            
            if img is not None:
                # img = img[:, 106 : 586]
                img_pil = Image.fromarray(img)
                img_pil.save('./images/' + self.workpiece.split('.')[0] + f"/projected_image_{x_crosssection}.png")
                plt.imshow(img, cmap='gray')
                plt.show()
            
        except Exception as e:
            print(f"处理过程中发生错误: {e}")


    def fill_point_cloud_section(self, image_path, output_path):
        # 1. 找到所有白色像素点的坐标
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image = self.unit_weldseam_pos(image)

        image[0, 4] = 255  # 添加

        white_pixels = np.where(image > 0)
        points = list(zip(white_pixels[1], white_pixels[0]))  # (x,y)坐标
        if not points:
            return image
        points.sort()  # 按x坐标排序
        
        # 2. 创建新图像
        height, width = image.shape
        filled_image = np.zeros_like(image)
        
        # 3. 使用插值连接点
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
        
        # 对所有x坐标进行遍历
        for x in range(width):
            try:
                # 找到对应的y值
                if x < min(x_coords) or x > max(x_coords):
                    continue
                    
                # 找到x左右两边最近的点
                mask = (x_coords <= x)
                if not any(mask) or all(mask):
                    continue
                    
                left_idx = np.where(mask)[0][-1]
                right_idx = np.where(~mask)[0][0]
                
                # 线性插值计算y值
                if left_idx == right_idx:
                    y = y_coords[left_idx]
                else:
                    x1, y1 = x_coords[left_idx], y_coords[left_idx]
                    x2, y2 = x_coords[right_idx], y_coords[right_idx]
                    # 检查除数是否为0
                    if x2 - x1 == 0:
                        y = y1
                    else:
                        y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                
                # 确保y值有效
                if np.isnan(y):
                    continue
                    
                y = int(np.clip(round(y), 0, height-1))
                filled_image[0:y, x] = 255  # 上方填充白色
                
            except Exception as e:
                print(f"Error at x={x}: {str(e)}")
                continue
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title("Original Image")
        # plt.imshow(image, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.title("Processed Image")
        # plt.imshow(filled_image, cmap='gray')
        # plt.show()
        cv2.imwrite(output_path, filled_image)
        # return filled_image

    @staticmethod
    def unit_weldseam_pos(image):
        points = np.where(image == 255)
        points = np.column_stack((points[1], points[0]))
        points = points[points[:, 0].argsort()]
        # print(points[-1])
        dx = image.shape[1] - points[-1][0]

        image = image[:, 0 : image.shape[1]  - (dx - 48)]
        pading = np.zeros((image.shape[0], (dx - 48)))
        image = np.hstack((pading, image))
        return image


if __name__ == "__main__":
    workpiece = "bead3.ply"
    x_crosssection = 8

    image_path = os.path.join("./images", workpiece.split('.')[0], f"projected_image_{x_crosssection}.png")  # 输入图像路径
    output_path = os.path.join("./images", workpiece.split('.')[0], f"processed_image.png")  # 输出图像路径


    processor = PointCloudProcessor(workpiece, x_crosssection, y1_crop=- 22.1, y2_crop=2.443, resolution=23 / 450, x_threshold=0.5, z_range=23)
    processor.process_point_cloud()

    processor.fill_point_cloud_section(image_path, output_path)

