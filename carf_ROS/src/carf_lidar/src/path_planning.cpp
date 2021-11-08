// BİSMİLLAHİRRAHMANİRRAHİM

#include<cmath>
#include <math.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <ros/ros.h>
#include<Eigen/Dense>
#include "pcl/point_cloud.h"
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include<pcl/filters/crop_box.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cv_bridge/cv_bridge.h>
#include <vector>
#include<image_transport/image_transport.h>
#include <pcl/filters/passthrough.h>
#include<tuple>
#include "std_msgs/Float32MultiArray.h"

# define M_PI  3.14159265358979323846 

// "İlim öğrenmek kadın-erkek herkese farzdır."

class pathCreator{
public:
    ros::Subscriber laser_sub;
    ros::Publisher laser_pub;
    ros::Publisher usbCom_pub;

    image_transport::Publisher image_pub;
    pathCreator(ros::NodeHandle &);
    void get_lidar_data(const sensor_msgs::PointCloud2ConstPtr&);
    std::tuple<float,float> get_vel_tetha(int goal_X, int goal_Y);

    const int map_size = 400;
    const int map_m_size = 12;
    float map_resolution = (float)map_m_size / (float)map_size; // in meters
    float minX, maxX, minY, maxY, minZ, maxZ;

};

// "İlim Çin'de bile olsa gidiniz."

pathCreator::pathCreator(ros::NodeHandle &nh){
    laser_sub = nh.subscribe("/velodyne_points", 10, &pathCreator::get_lidar_data, this);
    laser_pub = nh.advertise<sensor_msgs::PointCloud2>("/carf", 10);
    usbCom_pub = nh.advertise<std_msgs::Float32MultiArray>("/usb_com", 10);
    image_transport::ImageTransport it(nh);
    image_pub = it.advertise("/carf_mapImg", 10);

    minX =  0.0;
    minY = -10.0;
    maxX =  10.0;
    maxY =  10.0;
    minZ = -0.15;
    maxZ =  0.0;
}

// "Beşikten mezara kadar ilim öğreniniz."

std::tuple<float,float> pathCreator::get_vel_tetha(int goal_X, int goal_Y){
    float linear_vel = goal_X;
    float alpha = std::atan2(goal_Y, goal_X);
    return std::tuple<float, float>(linear_vel, alpha);
}

void pathCreator::get_lidar_data(const sensor_msgs::PointCloud2ConstPtr& input)
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*input, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> filter;
    filter.setInputCloud (cloud);
    filter.setFilterFieldName ("x");
    filter.setFilterLimits (minX, maxX);
    filter.filter (*cloudFiltered);
    filter.setInputCloud (cloudFiltered);
    filter.setFilterFieldName ("y");
    filter.setFilterLimits (minY, maxY);
    filter.filter (*cloudFiltered);
    filter.setInputCloud (cloudFiltered);
    filter.setFilterFieldName ("z");
    filter.setFilterLimits (minZ, maxZ);
    filter.filter (*cloudFiltered);

    sensor_msgs::PointCloud2 out;
    pcl::toROSMsg(*cloudFiltered, out);
    out.header = input->header;
    laser_pub.publish(out);

    int block = 1;
    int map2D[map_size][map_size]{};
    cv::Mat mapImage(map_size,map_size, CV_8UC3, cv::Scalar(255,255,255));
    cv::Vec3b block_color(0,0,0);
    int lidarX = (map_size/2), lidarY = (map_size/2);

    cv::circle(mapImage, cv::Point(lidarX, lidarY), 2, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_4);

    // "Hikmet(ilim) Müslümanın kayıp malıdır. Nerede bulursa alsın."

    int index_X, index_Y;
    for (int i = 0; i < cloudFiltered->size(); i++)
    {
        index_X = int((cloudFiltered->points[i].x*200) / map_m_size) + (map_size/2);
        index_Y = int((cloudFiltered->points[i].y*200) / map_m_size) + (map_size/2);

        if(index_X < 0 ||index_X > 399 ||index_Y < 0 ||index_Y > 399 ){

            std::cout << "x = " << index_X << " y= " << index_Y << std::endl;
            std::cout << "r_x = " << cloudFiltered->points[i].x << ", r_y = "
                      << cloudFiltered->points[i].y << std::endl;
        }

        mapImage.at<cv::Vec3b>(index_X,index_Y) = block_color;
        map2D[index_X][index_Y] = block;

    }

    //  "Muhakkak ki alimler, peygamberlerin mirasçılarıdır."

    int y1, y2, y;
    int path = 200;
    cv::Vec3b colorPath(255,0,0);

    std_msgs::Float32MultiArray steeringAngle;

    for (int i = 0; i < map_size; i++)
    {
        y1 = 0;
        y2 = 0;
        y = 0;

        for (int j = 0; j < map_size; j++)
        {
            if (map2D[i][j] == block){y1 = j; break;}
        }

        for (int k = map_size-1; k >=0; k--)
        {
            if (map2D[i][k] == block ){y2 = k; break;}
        }

        if (abs(y1 - y2) > 10)
        {
            y = int((y1 + y2) / 2);
            map2D[i][y] = path;
            mapImage.at<cv::Vec3b>(i,y) = colorPath;
        }
    }

    //  "Ya öğreten, ya öğrenen, ya dinleyen ya da ilmi seven ol. Fakat sakın beşincisi olma; (bunların dışında kalırsan) helâk olursun."
    
    int target_X_axis = 220; 
    int j, goal_X, goal_Y;
    cv::Vec3b colorGoal(0,255,0);
    float ld = 0.0, a_radian = 0.0, L = 1.8, angle = 0.0, back_y = 200.0, back_x = 130.0;

    // "İlim aramak için bir tarafa yönelen kimseye Allah, cennet yolunu kolaylaştırır."

    for (j = 0; j < map_size; j++)
    {
        if (map2D[target_X_axis][j] == path)
        {
            cv::circle(mapImage, cv::Point(j, target_X_axis), 2, colorGoal, cv::FILLED, cv::LINE_4);
            goal_X = target_X_axis - lidarX;
            goal_Y = j - lidarY;
            
            if ( goal_X < 20){
                std::cout << "HATA " << std::endl;
            }
            
            std::tuple<float, float> goal(get_vel_tetha(goal_X, goal_Y));
            float lin_vel = std::get<0>(goal) * (map_resolution);
            float alpha = std::get<1>(goal) * 180.0 / M_PI;
            
            //std::cout << "Ackerman_Linear_Vel: "<< lin_vel << " m/s, Ackerman_Degree: " <<  alpha << " deg" << std::endl;
            
            ld = sqrt( pow(((j - back_y)*map_resolution),2) + pow(((target_X_axis - back_x)*map_resolution),2));
            a_radian = std::atan2( ((j - back_y)), ((target_X_axis - back_x)) );
          
            angle = std::atan2((2 * L * std::sin(a_radian)) , ld);
            angle = angle * 180.0 / M_PI;

            //std::cout << "Pure_Pursuit_Steer_Angle = " << angle << std::endl;

            if ( angle == 0) {
                steeringAngle.data.push_back(0.01);
            }
            else {
                steeringAngle.data.push_back(alpha);
            }

            steeringAngle.data.push_back(lin_vel);
            usbCom_pub.publish(steeringAngle);
            
            break;
        }

    }  

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", mapImage).toImageMsg();
    image_pub.publish(msg);
}
 
// "İlmi öğrenip de başkalarına dağıtıp nakil etmeyen insan, altınları gömüp onu sarf etmeyen, ondan yedirip içirmeyen kimseye benzer."

int main(int argc, char **argv)  
{
    ros::init(argc,argv,"laser_node");
    ros::NodeHandle nh;
    pathCreator p(nh);
    ros::spin();
  
    return 0;  
}

//  "Hiç bilenlerle bilmeyenler bir olur mu?"