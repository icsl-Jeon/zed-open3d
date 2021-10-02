CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
PROJECT(ZED_OPEN3D)
set(CMAKE_BUILD_TYPE debug)


find_package(ZED 3 REQUIRED)
include_directories(${ZED_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})

find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")


find_package(Open3D )
if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
endif()
message(STATUS " open3d library: ${Open3D_LIBRARIES}")


include_directories(./include)
add_executable(DrawPointCloud test/DrawPointCloud.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(DrawPointCloud PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D )

add_executable(DrawImage test/DrawImage.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(DrawImage PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D )
