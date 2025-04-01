#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <mppi_rollouts/OdomProcessedFull2D.h>
#include "std_msgs/String.h"
#include <sstream>

class OdomProcessor
{
public:
    OdomProcessor()
    {
        // Initialize subscriber and publisher
        odom_sub_ = nh_.subscribe("/warty/odom", 10, &OdomProcessor::odomCallback, this);
        odom_pub_ = nh_.advertise<mppi_rollouts::OdomProcessedFull2D>("/warty/odom_processed_full2D", 10);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_;
    ros::Publisher odom_pub_;

    void odomCallback(const nav_msgs::Odometry::ConstPtr& odom)
    {
        // Debug
        ROS_INFO("Received odometry message at time: %f", odom->header.stamp.toSec());
        // ROS_INFO("Position message (x): %f", odom->pose.pose.position.x);
        
        // Declare a ROS message for the processed odometry. 
        mppi_rollouts::OdomProcessedFull2D processed_odom;

        // Copy the header from the odom message to preserve time stamps. 
        // Copy the header from the odom message to preserve time stamps. 
        // processed_odom.header.stamp = ros::Time::now(); // = odom->header;
        // processed_odom.header.frame_id = "warty/odom_processed";   // Parent frame
        // processed_odom.header.stamp = odom->header.stamp;
        // processed_odom.header.seq = odom->header.seq;
        // processed_odom.header.stamp = odom->header.stamp;
        // processed_odom.header.frame_id = odom->header.frame_id;


        // Set the time (use the time stamp from the odom topic)
        processed_odom.time = odom->header.stamp.toSec();
        
        // Copy the states. Convert quaternion to yaw angle. 
        processed_odom.xPos = odom->pose.pose.position.x;
        processed_odom.yPos = odom->pose.pose.position.y;
        processed_odom.yaw = tf::getYaw(odom->pose.pose.orientation);
        processed_odom.xVel = odom->twist.twist.linear.x;
        processed_odom.yVel = odom->twist.twist.linear.y;
        processed_odom.zAngVel = odom->twist.twist.angular.z;

        // Debug
        // ROS_INFO("Processed yaw message: %f", processed_odom.yaw);
        
        // Publish processed odometry
        odom_pub_.publish(processed_odom);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "odom_processor");
    OdomProcessor processor;
    ros::spin();
    return 0;
}