#ifndef CAFFE_NODEJS_UTIL_CONVERSION_H_
#define CAFFE_NODEJS_UTIL_CONVERSION_H_

#include <node_api.h>
#include <sstream>
using namespace std;
/**
* @desc: convert napi value into int
* @param env {napi_env}: enviroment variable
* @param n_value {napi_value}: candidate of conversation
* @param i_value {int}: convert into int
**/
inline int ConvertNapiToInt(napi_env env, napi_value* value, int* result){
  int32_t result32 = -1;
  NAPI_CALL(env, napi_get_value_int32(env, *value, &result32));
  *result = (int) result32;
  return *result;
}

/**
* @desc: convert int into napi value
* @param env {napi_env}: enviroment variable
* @param i_value {int*}: reference to candidate
* @param n_value {napi_value*}: reference to result
**/
inline napi_value ConvertIntToNapi(napi_env env, int* value, napi_value* result){
  NAPI_CALL(env, napi_create_uint32(env, (uint32_t)*value, result));
  return *result;
}




/**
* @desc: convert int into napi value
* @param env {napi_env}: enviroment variable
* @param value {float*}: reference to candidate
* @param result {napi_value*}: reference to result
**/
inline napi_value ConvertIntToNapi(napi_env env, float* value, napi_value* result){
  NAPI_CALL(env, napi_create_uint32(env, (uint32_t)*value, result));
  return *result;
}

#define INT_TO_NAPI(env, value, result) \
  if (!IntToNapi(env, value, result)) return NULL;

#define INT_TO_NAPI_RETVAL(env, value, result, retval) \
  if (!IntToNapi(env, value, result)) return retval;
/**
* @desc: convert int into napi value
* @param env {napi_env}: enviroment variable
* @param value {int*}: reference to candidate
* @param result {napi_value*}: reference to result
**/
inline napi_value IntToNapi(napi_env env, int* value, napi_value* result){
  NAPI_CALL(env, napi_create_uint32(env, (uint32_t)*value, result));
  return *result;
}



/**
* convert size_t to string
* @param *source: <*size_t>
* @param *result: <*std::string>
*/
inline std::string size_tTostring(size_t* source, std::string* result){
  std::stringstream ss;
  ss << source;
  *result = ss.str();
  return *result;
}


#endif // CAFFE_NODEJS_UTIL_CONVERSION_H_