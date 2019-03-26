# Install script for directory: /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/users/minesh.mathew/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libthpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libthpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libthpp.so"
         RPATH "$ORIGIN/../lib:/users/minesh.mathew/torch/install/lib:/users/minesh.mathew/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/libthpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libthpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libthpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libthpp.so"
         OLD_RPATH "/users/minesh.mathew/torch/install/lib:/users/minesh.mathew/local/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/users/minesh.mathew/torch/install/lib:/users/minesh.mathew/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libthpp.so")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/thpp" TYPE FILE FILES
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/Storage.h"
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/Storage-inl.h"
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/StorageSerialization-inl.h"
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/Tensor.h"
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/Tensor-inl.h"
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/TensorSerialization-inl.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/thpp/detail" TYPE FILE FILES
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/detail/Storage.h"
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/detail/StorageGeneric.h"
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/detail/Tensor.h"
    "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/detail/TensorGeneric.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/thpp/if/gen-cpp2" TYPE DIRECTORY FILES "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/thpp/if/gen-cpp2/" FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/[^/]*\\.tcc$")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/thpp/if" TYPE DIRECTORY FILES "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/if/" FILES_MATCHING REGEX "/[^/]*\\.thrift$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/googletest-release-1.7.0/cmake_install.cmake")
  include("/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
