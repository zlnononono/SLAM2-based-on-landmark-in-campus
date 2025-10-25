import cv2
import numpy as np
from ultralytics.solutions.pose_estimation import YOLOPoseEstimator

def demo_apollo_sequence(weights_path, sequence_path):
    """
    在Apollo数据集上进行演示
    
    Args:
        weights_path: YOLO权重文件路径
        sequence_path: Apollo序列数据路径
    """
    # 初始化位姿估计器
    estimator = YOLOPoseEstimator(
        yolo_weights_path=weights_path,
        feature_method='sift'  # 使用SIFT特征，通常比ORB更准确
    )
    
    # 设置Apollo数据集的相机参数
    # 注意：这里的参数需要根据实际的Apollo数据集相机标定结果修改
    estimator.camera_matrix = np.array([[2000, 0, 960],
                                      [0, 2000, 540],
                                      [0, 0, 1]], dtype=np.float32)
    
    # 读取图像序列
    import glob
    import os
    
    image_paths = sorted(glob.glob(os.path.join(sequence_path, '*.jpg')))
    images = [cv2.imread(path) for path in image_paths]
    
    # 处理序列
    poses = estimator.process_sequence(images, visualize=True)
    
    # 计算累积轨迹
    trajectory = []
    current_pose = np.eye(4)  # 4x4变换矩阵
    trajectory.append(current_pose[:3, 3])  # 保存位置
    
    for R, t in poses:
        # 构建变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t.reshape(3)
        
        # 更新当前位姿
        current_pose = current_pose @ transform
        trajectory.append(current_pose[:3, 3])
    
    # 保存轨迹结果
    trajectory = np.array(trajectory)
    np.save('trajectory.npy', trajectory)
    
    # 可视化轨迹
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def demo_realtime(weights_path, video_source=0):
    """
    实时视频演示
    
    Args:
        weights_path: YOLO权重文件路径
        video_source: 视频源
    """
    estimator = YOLOPoseEstimator(weights_path)
    estimator.process_realtime(video_source, visualize=True)

if __name__ == "__main__":
    # 使用预训练的YOLOv8模型
    weights_path = "yolov8n.pt"  # 或使用其他版本的权重
    
    # Apollo数据集演示
    print("Processing Apollo sequence...")
    demo_apollo_sequence(weights_path, "path/to/apollo/sequence")
    
    # 实时视频演示
    print("\nStarting realtime processing...")
    demo_realtime(weights_path) 