
打开两个终端,分别运行:

终端1:进入到/ORB_SLAM2/ultralytics-main:
(python detect_speedup_send.py +path to afterchanged dataset + weights ./weights/best.pt --conf 0.4 --save-txt --img-size 224)

终端2:进入到/ORB_SLAM2:
(./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml + path to dataset + path to associatefile)

