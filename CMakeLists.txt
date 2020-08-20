cmake_minimum_required(VERSION 2.8.3)
project(cumulative_cubic_B_spline)

set(CMAKE_VERBOSE_MAKEFILE "true")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_BUILD_TYPE "Release")
set(ADDITIONAL_CXX_FLAG "-Wall -O3 -g -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_CXX_FLAG}")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  rviz_visual_tools
  #image_transport
  #cv_bridge
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include/thirdparty/
  ${CERES_INCLUDE_DIRS}
)

#add_executable(cumulative_cubic_B_spline src/main.cpp src/utils.cpp src/ceres_solve.cpp src/data_structure.cpp)
#target_link_libraries(cumulative_cubic_B_spline ${catkin_LIBRARIES} ${CERES_LIBRARIES} dw)
#add_dependencies(cumulative_cubic_B_spline cumulative_cubic_B_spline_generate_messages_cpp)

add_executable(demo src/demo.cpp)
target_link_libraries(demo ${catkin_LIBRARIES} ${OpenCV_LIBS} dw)