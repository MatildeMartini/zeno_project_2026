######################################################################################
                                    MARTA_MSGS PKG
######################################################################################

This ROS package contains custom ROS message definitions required for operating with
Zeno AUV within a ROS environment. These message types allow to read all Zeno topics.

###################################  INSTALLATION  ###################################

To include 'marta_msgs' package in your ROS workspace, follow these steps:

1. Copy the 'marta_msgs' package into your workspace source directory:

your_workspace
|---src
    |--- marta_msgs
    |--- other_pkg 
    ...

2. Build your workspace:
cd ~/your_workspace
catkin_make


##############################  USAGE IN OTHER PACKAGES ##############################

In order to use marta_msgs in other packages, you must add 'marta_msgs' as a 
dependency following these steps:

1. In your other package go in 'package.xml' and add:

<build_depend>marta_msgs</build_depend>
<exec_depend>marta_msgs</exec_depend>
<exec_depend>message_runtime</exec_depend>   # To utilize custom generated msgs

2. In your other package go in 'CMakeList.txt', add the 'marta_msgs' to 
the following commands:

find_package(catkin REQUIRED COMPONENTS
    ...  (other dependencies here)
	message_runtime
    marta_msgs
)

generate_messages(
    DEPENDENCIES ... (other dependencies here) ... marta_msgs)

catkin_package(
    CATKIN_DEPENDS ... (other dependencies here) ... marta_msgs)

3. After these steps, rebuild your workspace to incorporate changes:

cd ~/your_workspace 
catkin_make

4. In order to utilize the desired message in your ROS node you must include them as
follows:

C++ 
#include marta_msgs/**NAME_CUSTOM_MSG**.h

python
from marta_msgs import **NAME_CUSTOM_MSG**

####################################  NAV STATUS  ####################################

One of the main messages included in the package is marta_msgs/NavStatus and it will 
be used to retrive the navigation information about Zeno. This message is published on 
the \nav_status topic that contains the vehicle's navigation status, estimated via 
the navigation filter. It includes the following useful fields:

    position:       expressed in geodetic coordinates (latitude and longitude [°]) and 
                    depth [m].

    orientation:    orientation in terms of Euler angles [rad].

    ned_speed:      linear speed [m/s] expressed in coordinates relative to the NED 
                    (North-East-Down) navigation frame.

    omega_body:     angular velocity [rad/s] expressed in coordinates relative to the 
                    vehicle-attached frame (body frame).

    gps_status:     indicates the GPS measurement fix; if the value is 1, the data 
                    is considered valid.

The ROS message published on this topic is a custom marta_msgs/NavStatus, whose 
structure is as follows:

NavStatus.msg

	Header header

	marta_msgs/Position     position
	marta_msgs/Euler        orientation
	marta_msgs/Quaternion   quaternion

	geometry_msgs/Vector3   ned_speed
	geometry_msgs/Vector3   omega_body

	uint8                   gps_status
	bool                    initialized


This message depends on others marta_msgs:

Position.msg
	float64 latitude
	float64 longitude
	float64 depth

Euler.msg
	float64 roll
	float64 pitch
	float64 yaw

Quaternion.msg
	float64 w
	float64 x
	float64 y
	float64 z