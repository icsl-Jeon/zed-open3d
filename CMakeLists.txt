CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
PROJECT(ZED_OPEN3D)

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


include_directories(./include)
add_executable(DrawPointCloud test/DrawPointCloud.cpp src/ZedUtils.cpp src/Misc.cpp)
target_link_libraries(DrawPointCloud ${ZED_LIBS} ${OpenCV_LIBRARIES})
