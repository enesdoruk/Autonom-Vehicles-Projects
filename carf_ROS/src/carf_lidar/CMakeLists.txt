cmake_minimum_required(VERSION 2.8.3)
project(carf_lidar)
 
 
find_package(catkin REQUIRED COMPONENTS
		rospy		
		roscpp
		pcl_ros
		sensor_msgs
		cv_bridge
		image_transport)
 
 
catkin_package()
 
include_directories( include ${catkin_INCLUDE_DIRS})

add_executable(laser_node src/path_planning.cpp)
target_link_libraries(laser_node ${catkin_LIBRARIES})
add_dependencies(laser_node carf_lidar_generate_messages_cpp)

catkin_install_python(PROGRAMS scripts/USBCom.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/tec_control.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/lidar_path.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
