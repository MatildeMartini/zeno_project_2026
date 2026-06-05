cmake_minimum_required(VERSION 2.8.3)
project(sss_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  marta_msgs
  cv_bridge
  message_generation
)


add_message_files(
  FILES
  GeolocatedObject.msg
  ImageMetadata.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  marta_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
  sensor_msgs
  marta_msgs
  cv_bridge
  message_runtime
)

catkin_install_python(PROGRAMS
  scripts/waterfall_creator_node.py
  scripts/object_classification_node.py
  scripts/geolocalization_node.py
  scripts/realtime_visualizer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
