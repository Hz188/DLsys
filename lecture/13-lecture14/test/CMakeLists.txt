cmake_minimum_required(VERSION 3.16)
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)

if (Python3_FOUND)
  message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
  message(STATUS "_Python3_EXECUTABLE: ${_Python3_EXECUTABLE}\n")
 
  message(STATUS "Python3_INCLUDE_DIR: ${Python3_INCLUDE_DIR}")
  message(STATUS "_Python3_INCLUDE_DIR: ${_Python3_INCLUDE_DIR}\n")
 
  message(STATUS "Python3_LIBRARY_DEBUG: ${Python3_LIBRARY_DEBUG}")
  message(STATUS "_Python3_LIBRARY_DEBUG: ${_Python3_LIBRARY_DEBUG}\n")
 
  message(STATUS "Python3_LIBRARY_RELEASE: ${Python3_LIBRARY_RELEASE}")
  message(STATUS "_Python3_LIBRARY_RELEASE: ${_Python3_LIBRARY_RELEASE}\n")
 
  message(STATUS "Python3_NumPy_INCLUDE_DIR: ${Python3_NumPy_INCLUDE_DIR}")
  message(STATUS "_Python3_NumPy_INCLUDE_DIR: ${_Python3_NumPy_INCLUDE_DIR}\n")
endif()
