#ifndef CAFFE_NODEJS_UTIL_CONVERSION_H_
#define CAFFE_NODEJS_UTIL_CONVERSION_H_

#include <node_api.h>

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
/*inline napi_value ConvertIntToNapi(napi_env env, int* value, napi_value* result){
  NAPI_CALL(env, napi_create_uint32(env, (uint32_t)*value, result));
  return *result;
}*/

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


#endif // CAFFE_NODEJS_UTIL_CONVERSION_H_