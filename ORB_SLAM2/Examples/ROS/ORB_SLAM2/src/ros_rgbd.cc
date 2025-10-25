/**
 * @file ros_rgbd.cc
 * @author guoqing (1337841346@qq.com)
 * @brief ORB RGB-D 输入的ROS节点实现
 * @version 0.1
 * @date 2019-08-06
 * 
 * @copyright Copyright (c) 2019
 * 
 */


/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include<opencv2/core/core.hpp>

#include"../../../include/System.h"
//for socket
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){}

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);

    ORB_SLAM2::System* mpSLAM;
};

void LoadBoundingBox(const string& strPathToDetectionResult, vector<std::pair<vector<double>, int>>& detect_result);

void LoadBoundingBoxFromPython(const string& resultFromPython, std::pair<vector<double>, int>& detect_result);
void MakeDetect_result(vector<std::pair<vector<double>, int>>& detect_result, int sockfd);

int main(int argc, char **argv)
{

    

    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    
    int sockfd;
	int len;
	struct sockaddr_un address;
	int result;
	int i,byte;
	char send_buf[128],ch_recv[1024];
 
	if((sockfd = socket(AF_UNIX, SOCK_STREAM, 0))==-1)//创建socket，指定通信协议为AF_UNIX,数据方式SOCK_STREAM
	{
		perror("socket");
		exit(EXIT_FAILURE);
	}
	
	//配置server_address
	address.sun_family = AF_UNIX;
	strcpy(address.sun_path, "/home/cjj/server_socket");
	len = sizeof(address);

 
	result = connect(sockfd, (struct sockaddr *)&address, len);
 
	if(result == -1) 
	{
		printf("ensure the server is up\n");
        	perror("connect");
        	exit(EXIT_FAILURE);
    }

    vector<std::pair<vector<double>, int>> detect_result,detect_result_test2;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/depth_registered/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD,vector<std::pair<vector<double>, int>>& detect_result , int sockfd)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    MakeDetect_result(detect_result,sockfd);

    mpSLAM->TrackRGBD(cv_ptrRGB->image,cv_ptrD->image,cv_ptrRGB->header.stamp.toSec());
}

void LoadBoundingBox(const string& strPathToDetectionResult, vector<std::pair<vector<double>, int>>& detect_result){
    ifstream infile;
    infile.open(strPathToDetectionResult);
    if (!infile.is_open()) {
        cout<<"yolo_detection file open fail"<<endl;
        exit(233);
    }
    vector<double> result_parameter;
    string line;
    while (getline(infile, line)){
        int sum = 0, num_bit = 0;
        for (char c : line) {//读取数字.    例如读取"748",先读7,再7*10+8=78,再78*10+4,最后读到空格结束
            if (c >= '0' && c <= '9') {
                num_bit = c - '0';
                sum = sum * 10 + num_bit;
            } else if (c == ' ') {
                result_parameter.push_back(sum);
                sum = 0;
                num_bit = 0;
            }
        }

        string idx_begin = "class:";//读取物体类别
        int idx = line.find(idx_begin);
        string idx_end = "0.";
        int idx2 = line.find(idx_end);
        string class_label;
        for (int j = idx + 6; j < idx2-1; ++j){
            class_label += line[j];
        }
        // cout << "**" << class_label << "**";

        int class_id = -1;//存入识别物体的种类
        if (class_label == "person") { //高动态物体:人,动物等
            class_id = 3;
        }

        if (class_label == "tv" ||   //低动态物体(在程序中可以假设为一直静态的物体):tv,refrigerator
            class_label == "refrigerator" || 
            class_label == "teddy bear") {
            class_id = 1;
        }

        if (class_label == "chair" || //中动态物体,在程序中不做先验动态静态判断
            class_label == "car"){
            class_id =2;
        }

        detect_result.emplace_back(result_parameter,class_id);
        result_parameter.clear();
        line.clear();
    }
    infile.close();

}

//在一句话中提取出四个边框值和物体类别,such as: left:1 top:134 right:269 bottom:478 class:person 0.79
void LoadBoundingBoxFromPython(const string& resultFromPython, std::pair<vector<double>, int>& detect_result){
    
    if(resultFromPython.empty())
    {
        cerr << "no string from python! " << endl;
    }
    // cout << "here is LoadBoundingBoxFromPython " << endl;
    vector<double> result_parameter;
    int sum = 0, num_bit = 0;

    for (char c : resultFromPython) {//读取数字.    例如读取"748",先读7,再7*10+8=78,再78*10+4,最后读到空格结束
        if (c >= '0' && c <= '9') {
            num_bit = c - '0';
            sum = sum * 10 + num_bit;
        } else if (c == ' ') {
            result_parameter.push_back(sum);
            sum = 0;
            num_bit = 0;
        }
    }

    detect_result.first = result_parameter;
    // cout << "detect_result.first size is : " << detect_result.first.size() << endl;

    string idx_begin = "class:";//读取物体类别
    int idx = resultFromPython.find(idx_begin);
    string idx_end = "0.";
    int idx2 = resultFromPython.find(idx_end);
    string class_label;
    for (int j = idx + 6; j < idx2-1; ++j){
        class_label += resultFromPython[j];
    }

    int class_id = -1;//存入识别物体的种类

    if (class_label == "tv" ||   //低动态物体(在程序中可以假设为一直静态的物体):tv,refrigerator
        class_label == "refrigerator" || 
        class_label == "teddy bear"||
        class_label == "laptop") {
        class_id = 1;
    }

    if (class_label == "chair" || //中动态物体,在程序中不做先验动态静态判断
        class_label == "car"){
        class_id =2;
    } 

    if (class_label == "person") { //高动态物体:人,动物等
        class_id = 3;
    }

    detect_result.second = class_id;
    cout << "LoadBoundingBoxFromPython class id is: " << class_id << endl;

}

//通过UNIX的协议,从python进程中获取一帧图像的物体框
void MakeDetect_result(vector<std::pair<vector<double>, int>>& detect_result , int sockfd){
    detect_result.clear();

	std::pair<vector<double>, int> detect_result_str;
    int byte;
	char send_buf[128],ch_recv[1024];

    sprintf(send_buf,"ok");//用sprintf事先把消息写到send_buf
	if((byte=write(sockfd, send_buf, sizeof(send_buf)))==-1)
    {
		perror("write");
		exit(EXIT_FAILURE);
	}

    if((byte=read(sockfd,&ch_recv,1000))==-1)
	{
		perror("read");
		exit(EXIT_FAILURE);
	}
    // cout << "**ch_recv is : \n" << ch_recv << endl;

    char *ptr;//char[]可读可写,可以修改字符串的内容。char*可读不可写，写入就会导致段错误
    ptr = strtok(ch_recv, "*");//字符串分割函数
    //检查ptr是否全部为string类型
    
    
 
    while(ptr != NULL){
        printf("ptr=%s\n",ptr);
//         cout << "**ch_recv is : \n" << ch_recv << endl;
// for (size_t i = 0; i < strlen(ptr); i++)
//     {
//         //检查prt[i]是否为乱码
//             cout << "ptr[" << i << "] is : " << ptr[i] << endl; 
//     }
        if ( strlen(ptr)>20 ){//试图去除乱码,乱码原因未知...好像并不能去除,留着吧,心理安慰下
            // cout << strlen(ptr) << endl;
            string ptr_str = ptr;
            LoadBoundingBoxFromPython(ptr_str,detect_result_str);
        } 

        detect_result.emplace_back(detect_result_str);
        // cout << "hh: " << ptr_str << endl;  
        ptr = strtok(NULL, "*");
    }
    // cout << "detect_result size is : " << detect_result.size() << endl;
    // for (int k=0; k<detect_result.size(); ++k)
        // cout << "detect_result is : \n " << detect_result[k].second << endl;


}



