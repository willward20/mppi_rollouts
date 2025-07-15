#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_datatypes.h>
#include <mppi_rollouts/OdomCmdVelProcessedFull2D.h>
#include "std_msgs/String.h"
#include <sstream>
#include <mutex>

class OdomCmdVelProcessor
{
public:
    OdomCmdVelProcessor()
    {
        // Initialize subscriber and publisher
        odom_sub_ = nh_.subscribe("/warty/odom/throttled", 10, &OdomCmdVelProcessor::odomCallback, this);
        cmd_vel_sub_ = nh_.subscribe("/warty/cmd_vel/throttled", 10, &OdomCmdVelProcessor::cmdVelCallback, this);
        odom_cmd_vel_pub_ = nh_.advertise<mppi_rollouts::OdomCmdVelProcessedFull2D>("/warty/odom_cmd_vel_processed_full2D", 10);
        
        // Initialize command velocity
        latest_cmd_vel_.linear.x = 0.0;
        latest_cmd_vel_.angular.z = 0.0;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_;
    ros::Subscriber cmd_vel_sub_;
    ros::Publisher odom_cmd_vel_pub_;

    geometry_msgs::Twist latest_cmd_vel_;
    std::mutex cmd_mutex_;  // Protects access to latest_cmd_vel_

    void odomCallback(const nav_msgs::Odometry::ConstPtr& odom)
    {
        // Debug
        ROS_INFO("Received odometry message at time: %f", odom->header.stamp.toSec());
        
        // Declare a ROS message for the processed odometry. 
        mppi_rollouts::OdomCmdVelProcessedFull2D processed_odom_cmd_vel;

        // Set the time (use the time stamp from the odom topic)
        processed_odom_cmd_vel.time = odom->header.stamp.toSec();
        
        // Copy the states. Convert quaternion to yaw angle. 
        processed_odom_cmd_vel.xPos = odom->pose.pose.position.x;
        processed_odom_cmd_vel.yPos = odom->pose.pose.position.y;
        processed_odom_cmd_vel.yaw = tf::getYaw(odom->pose.pose.orientation);
        processed_odom_cmd_vel.xVel = odom->twist.twist.linear.x;
        processed_odom_cmd_vel.yVel = odom->twist.twist.linear.y;
        processed_odom_cmd_vel.zAngVel = odom->twist.twist.angular.z;

        // Include latest cmd_vel values
        {
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            processed_odom_cmd_vel.cmd_xVel = latest_cmd_vel_.linear.x;
            processed_odom_cmd_vel.cmd_zAngVel = latest_cmd_vel_.angular.z;
        }
        
        // Publish processed odometry and cmd_vel
        odom_cmd_vel_pub_.publish(processed_odom_cmd_vel);
    }

    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        latest_cmd_vel_ = *msg;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "odom_cmd_vel_processor");
    OdomCmdVelProcessor processor;
    ros::spin();
    return 0;
}