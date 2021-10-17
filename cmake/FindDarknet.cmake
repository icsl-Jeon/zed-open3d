#
#
# FindDarknet.cmake
#
#	Find needed variables for including Darknet lib to a project
#	Variables:
#		- Darknet_FOUND
#		- Darknet_INCLUDE_DIRS
#		- Darknet_LIBRARIES
#
#


find_path(Darknet_BASE_DIR "include/darknet/darknet.h"
        PATHS "/usr/local"
        "/usr")

if(Darknet_BASE_DIR)
    set(Darknet_INCLUDE_DIRS ${Darknet_BASE_DIR}/include)
    mark_as_advanced(Darknet_INCLUDE_DIRS)

    # Libraries
    set(Darknet_SO libdarknet.so)
    set(Darknet_A libdarknet.a)

    find_library(Darknet_SO_FOUND
            NAMES ${Darknet_SO}
            PATHS 	"/usr/lib/"
            "/usr/local/lib")

    find_library(Darknet_A_FOUND
            NAMES ${Darknet_A}
            PATHS 	"/usr/lib/"
            "/usr/local/lib")

    set(Darknet_LIBRARIES  ${Darknet_SO_FOUND})

    mark_as_advanced(Darknet_LIBRARIES)
endif(Darknet_BASE_DIR)