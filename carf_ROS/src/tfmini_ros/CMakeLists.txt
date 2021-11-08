cmake_minimum_required(VERSION 2.8.3)
project(tfmini_ros)

 find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  roscpp
)

 find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
  INCLUDE_DIRS include
)

 include_directories(${catkin_INCLUDE_DIRS} include)

 add_executable(tfmini_ros_front
   src/TFmini_front.cpp
   src/TFmini.cpp
 )

 add_executable(tfmini_ros_left
   src/TFmini_left.cpp
   src/TFmini.cpp
 )

 add_executable(tfmini_ros_right
   src/TFmini_right.cpp
   src/TFmini.cpp
 )
 target_link_libraries(tfmini_ros_front
   ${catkin_LIBRARIES}
 )

 target_link_libraries(tfmini_ros_left
   ${catkin_LIBRARIES}
 ) 

 target_link_libraries(tfmini_ros_right
   ${catkin_LIBRARIES}
 )
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=gnu++11")

# install(TARGETS tfmini_ros_node tfmini_ros_front
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(TARGETS tfmini_ros_node tfmini_ros_left
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
# install(DIRECTORY include
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
# )

# install(FILES
#  launch/tfmini.launch
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

