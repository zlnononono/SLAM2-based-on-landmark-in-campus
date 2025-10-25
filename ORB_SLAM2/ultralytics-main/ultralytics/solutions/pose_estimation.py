import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    """YOLO目标检测器，只负责检测并输出目标区域"""
    
    def __init__(self, weights_path='yolov8n.pt'):
        self.model = YOLO(weights_path)
        
    def detect(self, image):
        """
        检测图像中的目标
        
        Args:
            image: 输入图像
            
        Returns:
            list: 检测框列表 [(x1,y1,x2,y2), ...]
        """
        results = self.model(image, verbose=False)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes

class ORBSLAMFrontend:
    """ORB-SLAM2前端，负责特征提取和匹配"""
    
    def __init__(self):
        # 使用ORB-SLAM2的参数配置
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # 特征匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 相机参数
        self.camera_matrix = np.array([[1000, 0, 640],
                                     [0, 1000, 480],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1))
        
    def extract_features(self, image):
        """提取ORB特征点和描述子"""
        keypoints = self.orb.detect(image, None)
        keypoints, descriptors = self.orb.compute(image, keypoints)
        return keypoints, descriptors
        
    def is_point_in_regions(self, point, regions):
        """判断点是否在目标区域内"""
        x, y = point
        for x1, y1, x2, y2 in regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False
        
    def match_with_priority(self, kp1, des1, regions1, kp2, des2, regions2):
        """
        特征匹配，优先考虑目标区域内的点
        
        Args:
            kp1, des1: 第一帧的特征点和描述子
            regions1: 第一帧的目标区域
            kp2, des2: 第二帧的特征点和描述子
            regions2: 第二帧的目标区域
            
        Returns:
            tuple: (pts1, pts2) 匹配点对的坐标
        """
        if des1 is None or des2 is None:
            return None, None
            
        # 基本匹配
        matches = self.matcher.match(des1, des2)
        
        # 计算每个匹配的优先级
        priority_matches = []
        for m in matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            
            # 判断是否在目标区域内
            in_region1 = self.is_point_in_regions(pt1, regions1)
            in_region2 = self.is_point_in_regions(pt2, regions2)
            
            # 计算优先级分数
            score = 2 if (in_region1 and in_region2) else \
                   1 if (in_region1 or in_region2) else 0
                   
            priority_matches.append((score, m.distance, m))
            
        # 根据优先级和距离排序
        priority_matches.sort(key=lambda x: (-x[0], x[1]))
        
        # 选择最佳匹配点对
        pts1 = []
        pts2 = []
        used_query = set()
        used_train = set()
        
        for score, dist, m in priority_matches:
            if len(pts1) >= 100:  # 限制匹配点数量
                break
                
            if m.queryIdx not in used_query and m.trainIdx not in used_train:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
                used_query.add(m.queryIdx)
                used_train.add(m.trainIdx)
                
        return np.float32(pts1), np.float32(pts2)
        
    def estimate_pose(self, pts1, pts2):
        """使用ORB-SLAM2的方法估计位姿"""
        if len(pts1) < 8:  # 最少需要8个点
            return None, None
            
        # 计算本质矩阵
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            return None, None
            
        # 从本质矩阵恢复位姿
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        return R, t

class YOLOORBSLAM:
    """YOLO与ORB-SLAM2集成系统"""
    
    def __init__(self, yolo_weights='yolov8n.pt'):
        self.detector = YOLODetector(yolo_weights)
        self.frontend = ORBSLAMFrontend()
        
    def process_frame(self, frame1, frame2, visualize=False):
        """
        处理两帧图像
        
        Args:
            frame1, frame2: 两帧图像
            visualize: 是否可视化结果
            
        Returns:
            tuple: (R, t) 旋转矩阵和平移向量
        """
        # 1. YOLO目标检测
        regions1 = self.detector.detect(frame1)
        regions2 = self.detector.detect(frame2)
        
        # 2. 特征提取
        kp1, des1 = self.frontend.extract_features(frame1)
        kp2, des2 = self.frontend.extract_features(frame2)
        
        # 3. 优先特征匹配
        pts1, pts2 = self.frontend.match_with_priority(
            kp1, des1, regions1,
            kp2, des2, regions2
        )
        
        # 4. 位姿估计
        R, t = self.frontend.estimate_pose(pts1, pts2)
        
        # 可视化
        if visualize and pts1 is not None:
            self.visualize(frame1, frame2, regions1, regions2, pts1, pts2)
            
        return R, t
        
    def visualize(self, frame1, frame2, regions1, regions2, pts1, pts2):
        """可视化检测和匹配结果"""
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = frame1
        vis_img[:h2, w1:w1+w2] = frame2
        
        # 绘制检测框
        for x1, y1, x2, y2 in regions1:
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x1, y1, x2, y2 in regions2:
            cv2.rectangle(vis_img, (x1 + w1, y1), (x2 + w1, y2), (0, 255, 0), 2)
            
        # 绘制匹配线
        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            x1, y1 = map(int, [x1, y1])
            x2, y2 = map(int, [x2 + w1, y2])
            
            # 判断是否在目标区域内
            in_region1 = self.frontend.is_point_in_regions((x1, y1), regions1)
            in_region2 = self.frontend.is_point_in_regions((x2 - w1, y2), regions2)
            
            if in_region1 and in_region2:
                color = (0, 255, 0)  # 绿色：两个点都在目标区域内
            elif in_region1 or in_region2:
                color = (255, 255, 0)  # 黄色：一个点在目标区域内
            else:
                color = (0, 0, 255)  # 红色：都不在目标区域内
                
            cv2.line(vis_img, (x1, y1), (x2, y2), color, 1)
            cv2.circle(vis_img, (x1, y1), 3, color, -1)
            cv2.circle(vis_img, (x2, y2), 3, color, -1)
            
        cv2.imshow('YOLO + ORB-SLAM2', vis_img)
        cv2.waitKey(1)
        
    def process_sequence(self, image_sequence, visualize=False):
        """处理图像序列"""
        poses = []
        prev_frame = None
        
        for frame in image_sequence:
            if prev_frame is not None:
                R, t = self.process_frame(prev_frame, frame, visualize)
                if R is not None and t is not None:
                    poses.append((R, t))
            prev_frame = frame.copy()
            
        return poses
        
    def process_realtime(self, video_source=0, visualize=True):
        """实时处理视频流"""
        cap = cv2.VideoCapture(video_source)
        prev_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is not None:
                R, t = self.process_frame(prev_frame, frame, visualize)
                if R is not None and t is not None:
                    print(f"Rotation:\n{R}\nTranslation:\n{t}")
                    
            prev_frame = frame.copy()
            
            if visualize:
                cv2.imshow('Input', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        cv2.destroyAllWindows() 