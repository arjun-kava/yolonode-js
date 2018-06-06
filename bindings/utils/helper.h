#ifndef CAFFE_NODEJS_UTIL_HELPER_H_
#define CAFFE_NODEJS_UTIL_HELPER_H_

#include <node_api.h>
#include <vector>
#include <string>

#include <stdio.h>  /* defines FILENAME_MAX */
// #define WINDOWS  /* uncomment this line to use it for windows.*/ 
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include<iostream>

#define DECLARE_NAPI_PROPERTY(name, func) \
  { (name), 0, (func), 0, 0, 0, napi_default, 0 }

#define ARRAY_SIZE(array) (sizeof(array) / sizeof(array[0]))

inline std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}


#endif // CAFFE_NODEJS_UTIL_HELPER_H_

