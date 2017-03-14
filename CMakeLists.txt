CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
project(Vision2017)

SET(EXEC_NAME Vision2017)

SET(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)  
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/bin")

IF(WIN32) # Windows
find_package(CUDA 7.0 REQUIRED)
ELSE() # Linux
find_package(ZED REQUIRED)
find_package(CUDA 6.5 REQUIRED)
ENDIF(WIN32)
find_package(OpenCV 2.4 COMPONENTS core highgui imgproc features2d gpu REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")
FILE(GLOB_RECURSE SRC_CU_FOLDER "${SRC_FOLDER}/*.cu")


#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_52)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

cuda_add_executable(${EXEC_NAME} ${SRC_FILES} ${SRC_CU_FOLDER})
cuda_add_executable(${EXEC_NAME}-test ${SRC_FILES} ${SRC_CU_FOLDER})
cuda_add_executable(${EXEC_NAME}-gear-test ${SRC_FILES} ${SRC_CU_FOLDER})
cuda_add_executable(${EXEC_NAME}-gear ${SRC_FILES} ${SRC_CU_FOLDER})
cuda_add_executable(${EXEC_NAME}-gear2-test ${SRC_FILES} ${SRC_CU_FOLDER})
cuda_add_executable(${EXEC_NAME}-gear2 ${SRC_FILES} ${SRC_CU_FOLDER})

SET_TARGET_PROPERTIES(${EXEC_NAME}-test PROPERTIES COMPILE_FLAGS "-DSHOW")
SET_TARGET_PROPERTIES(${EXEC_NAME}-gear PROPERTIES COMPILE_FLAGS "-DGEAR")
SET_TARGET_PROPERTIES(${EXEC_NAME}-gear-test PROPERTIES COMPILE_FLAGS "-DGEAR -DSHOW")
SET_TARGET_PROPERTIES(${EXEC_NAME}-gear2 PROPERTIES COMPILE_FLAGS "-DGEAR2")
SET_TARGET_PROPERTIES(${EXEC_NAME}-gear2-test PROPERTIES COMPILE_FLAGS "-DGEAR2 -DSHOW")


add_definitions(-std=c++11)

TARGET_LINK_LIBRARIES(${EXEC_NAME}
                        ${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY}
			)

TARGET_LINK_LIBRARIES(${EXEC_NAME}-test
                        ${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY}
			)

TARGET_LINK_LIBRARIES(${EXEC_NAME}-gear-test
                        ${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY}
			)

TARGET_LINK_LIBRARIES(${EXEC_NAME}-gear
                        ${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY}
			)

TARGET_LINK_LIBRARIES(${EXEC_NAME}-gear2-test
                        ${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY}
			)

TARGET_LINK_LIBRARIES(${EXEC_NAME}-gear2
                        ${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY}
			)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3" )
