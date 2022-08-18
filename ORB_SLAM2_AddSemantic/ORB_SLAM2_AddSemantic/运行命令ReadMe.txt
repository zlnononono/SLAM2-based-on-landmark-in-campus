2021-4-30----new!
#添加了python-c++通信:yolo和orb的tracking线程并行,orb在i5 cpu运行需要0.0282526一帧,所以yolo只要在0.02s内即可
因此可以使用yolov5s+imgsize224-320-416; yolov5l+imgsize224; yolov5m+imgsize224
打开两个终端,分别运行:

终端1:进入到/ORB_SLAM2_AddSemantic/yolov5_RemoveDynamic:
(python detect_speedup_send.py +path to afterchanged dataset + weights ./weights/yolov5s.pt --conf 0.4 --save-txt --img-size 224)
一个例子:
python detect_speedup_send.py --source /home/jy/Desktop/dataset/TUM/tum_fr3_walking_xyz_afterchange/rgb/ --weights ./weights/yolov5s.pt --conf 0.4 --save-txt --img-size 224


终端2:进入到/ORB_SLAM2_AddSemantic:
(./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml + path to dataset + path to associatefile)
一个例子:
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ~/Desktop/dataset/TUM/rgbd_dataset_freiburg3_walking_xyz ~/Desktop/dataset/TUM/rgbd_dataset_freiburg3_walking_xyz/associate.txt
