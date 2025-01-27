import cv2
import numpy as np
import matplotlib.pyplot as plt

def fit_line_and_process_image(image_path, output_path):
    # 加载图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 计算每一列的白色像素数量
    white_pixel_counts = np.sum(image == 255, axis=0)
    
    # 找到白色像素最多的列
    max_white_column = np.argmax(white_pixel_counts)
    print(f"白色像素最多的列: {max_white_column}")
    
    # 找到350-500列之间的白色像素
    mask = (image[:, 100:200] == 255)
    y_indices, x_indices = np.where(mask)
    x_indices += 100  # 调整x_indices到原始图像的列索引

    # 使用最小二乘法拟合直线
    if len(x_indices) > 0:
        A = np.vstack([x_indices, np.ones(len(x_indices))]).T
        m, c = np.linalg.lstsq(A, y_indices, rcond=None)[0]
        print(f"拟合直线方程: y = {m:.2f}x + {c:.2f}")

        # 检查是否存在 NaN 值
        if np.isnan(m) or np.isnan(c):
            print("拟合直线过程中出现 NaN 值！")
            return

        # 创建一个新的图像
        processed_image = np.zeros_like(image)

        # 根据拟合的直线处理图像
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if y < m * x + c:
                    processed_image[y, x] = 255  # 左边的像素设为白色
                else:
                    processed_image[y, x] = 0  # 右边的像素设为黑色

        # 将该列右侧的所有像素设为黑色
        processed_image[:, max_white_column + 1:] = 0

        # 保存处理后的图像
        cv2.imwrite(output_path, processed_image)

        # 显示原始图像和处理后的图像
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Processed Image")
        plt.imshow(processed_image, cmap='gray')
        plt.show()
    else:
        print("在指定范围内没有找到白色像素！")




def fill_point_cloud_section(image_path, output_path):
    # 1. 找到所有白色像素点的坐标
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = unit_weldseam_pos(image)

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
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(filled_image, cmap='gray')
    plt.show()
    cv2.imwrite(output_path, filled_image)
    # return filled_image

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
    image_path = "./images/bead3/projected_image_8.png"  # 输入图像路径
    output_path = "./images/bead3/processed_image.png"  # 输出图像路径
    # fit_line_and_process_image(image_path, output_path)
    fill_point_cloud_section(image_path, output_path)
