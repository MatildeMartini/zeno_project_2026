cmake_minimum_required(VERSION 2.8.3)
project(marta_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs geometry_msgs sensor_msgs message_generation)


## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg)

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs  
   geometry_msgs
   sensor_msgs
 )

catkin_package()