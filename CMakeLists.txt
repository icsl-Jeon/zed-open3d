CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
PROJECT(ZED_OPEN3D)
set(CMAKE_BUILD_TYPE debug)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
message("${CMAKE_MODULE_PATH}")

# For linking darknet (TODO: FindDarknet).
set(DARKNET_PATH /home/jbs/catkin_ws/src/darknet)
set(DARKNET_LIB libdarknet)
include_directories(${DARKNET_PATH}/include)

add_library(${DARKNET_LIB} SHARED IMPORTED)
set_property(
        TARGET ${DARKNET_LIB}
        PROPERTY IMPORTED_LOCATION
        "${DARKNET_PATH}/libdarknet.so"
)

# TODO
#find_package(Darknet)
#message("darknet: ${Darknet_INCLUDE_DIRS}")
#include_directories(${Darknet_INCLUDE_DIRS})

find_package(ZED 3 REQUIRED)
include_directories(${ZED_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})

find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})
message("cuda = ${CUDA_LIBRARIES}")

SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

find_package(OpenCV 4.5.3 REQUIRED)
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

add_executable(ImageCPU test/ImageCPU.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(ImageCPU PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES})

add_executable(ImageGPU test/ImageGPU.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(ImageGPU PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES})

add_executable(ImageGPU_ops test/ImageGPU_ops.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(ImageGPU_ops PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES})

add_executable(PointCloudGPU test/PointCloudGPU.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(PointCloudGPU PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES})

add_executable(PointCloudCPU test/PointCloudCPU.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(PointCloudCPU PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES})

add_executable(HumanAttention test/HumanAttention.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(HumanAttention PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES})

add_executable(DrawAttention test/DrawAttention.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(DrawAttention PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D )

add_executable(OoiExtractor test/OoiExtractor.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(OoiExtractor PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES} ${DARKNET_LIB} )

add_executable(TsdfIntegration test/TsdfIntegration.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(TsdfIntegration PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES} ${DARKNET_LIB} )

add_executable(OoiExtractorMesh test/OoiExtractorMesh.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp )
target_link_libraries(OoiExtractorMesh PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES} ${DARKNET_LIB} )

add_executable(RunScene exe/RunScene.cpp src/ZedUtils.cpp src/Misc.cpp src/Open3dUtils.cpp src/SceneInterpreter.cpp)
target_link_libraries(RunScene PRIVATE  ${ZED_LIBS}   ${OpenCV_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES} ${DARKNET_LIB} )




