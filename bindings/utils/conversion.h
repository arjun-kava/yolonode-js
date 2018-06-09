#ifndef CAFFE_NODEJS_UTIL_CONVERSION_H_
#define CAFFE_NODEJS_UTIL_CONVERSION_H_

#include <node_api.h>
#include <sstream>
#include <string.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
/**
* @desc: convert napi value into int
* @param env {napi_env}: enviroment variable
* @param n_value {napi_value}: candidate of conversation
* @param i_value {int}: convert into int
**/
#define NAPI_TO_INT(env, value, result) NapiToInt(env, value, result);
#define NAPI_TO_INT_RETVAL(env, value, result, retval) \
  if (!NapiToInt(env, value, result)) return retval;
inline int NapiToInt(napi_env env, napi_value* value, int* result){
  int32_t result32 = -1;
  NAPI_CALL(env, napi_get_value_int32(env, *value, &result32));
  *result = (int) result32;
  return *result;
}

/**
* @desc: convert int into napi value
* @param env {napi_env}: enviroment variable
* @param value {int*}: reference to candidate
* @param result {napi_value*}: reference to result
**/
#define INT_TO_NAPI(env, value, result) \
  if (!IntToNapi(env, value, result)) return NULL;
#define INT_TO_NAPI_RETVAL(env, value, result, retval) \
  if (!IntToNapi(env, value, result)) return retval;
inline napi_value IntToNapi(napi_env env, int* value, napi_value* result){
  NAPI_CALL(env, napi_create_uint32(env, (uint32_t)*value, result));
  return *result;
}

/**
* @desc: convert int into napi value
* @param env {napi_env}: enviroment variable
* @param value {int*}: reference to candidate
* @param result {napi_value*}: reference to result
**/
#define FLOAT_TO_NAPI(env, value, result) FloatToNapi(env, value, result);
#define FLOAT_TO_NAPI_RETVAL(env, value, result, retval) \
  FloatToNapi(env, value, result) \
  return retval;
inline napi_value FloatToNapi(napi_env env, float* value, napi_value* result){
  NAPI_CALL(env, napi_create_int64(env, (int64_t)*value, result));
  return *result;
}

/**
* @desc: convert napi value to float
* @param env: {napi_env} 
* @param source: {*napi_value} 
* @param result: {*float} 
*/
#define NAPI_TO_FLOAT(env, source, result) NapiTofloat(env, source, result);
#define NAPI_TO_FLOAT_RETVAL(env, source, result, retval) \
  NapiTofloat(env, source, result) \
  return retval;
inline float NapiTofloat(napi_env env, napi_value* source, float* result){
  int64_t value;
  NAPI_CALL(env, napi_get_value_int64(env, *source, &value));
  *result = value;
  return *result;
}

/**
* @desc: convert char array to napi string
*/
#define CHAR_TO_NAPI(env, source, result) CharToNapi(env, source, result);
#define CHAR_TO_NAPI_RETVAL(env, source, result, retval) \
  CharToNapi(env, source, result) \
  return retval;
inline napi_value CharToNapi(napi_env env,const char* source, napi_value* target) {
  size_t length= strlen(source);
  NAPI_CALL(env, napi_create_string_utf8(env, source, length, target));
  return *target;
}

/**
* @description: convert napi string to char* array
*/
#define NAPI_TO_CHAR(env, source, target, length) NapiToChar(env, source, target, length);
#define NAPI_TO_CHAR_RETVAL(env, source, target, length, retval) \
  NapiToChar(env, source, target, length) \
  return retval;
inline char NapiToChar(napi_env env,napi_value* source,char* target, size_t *length){
    NAPI_CALL(env, napi_get_value_string_utf8(env, *source, target, *length, NULL));
    return *target;
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


/**
* convert float array to js array
* @param env: <napi_env> 
* @param source: <*float> 
* @param target: <*napi_value> 
*/
#define FLOAT_MARRAY_TO_NAPI(env, source, rows, cols, target) FloatMArrayToNapi(env, source, rows, cols, target);
#define FLOAT_MARRAY_TO_NAPI_RETVAL(env, source, rows, cols, target, retval) \
  FloatMArrayToNapi(env, source, rows, cols, target) \
  return retval;
inline static napi_value FloatMArrayToNapi(napi_env env,float** source, int* rows,int* cols, napi_value* target){
    NAPI_CALL(env, napi_create_array(env, target));
   
    for (int i=0; i<*rows; i++)
    {
        napi_value dimension;
        NAPI_CALL(env, napi_create_array(env, &dimension));
        for(int j=0; j<*cols; j++)
        {
            napi_value elem;
            NAPI_CALL(env, napi_create_int64(env, (int64_t) source[i][j], &elem));
            NAPI_CALL(env, napi_set_element(env, dimension, j, elem));
        }
        NAPI_CALL(env, napi_set_element(env, *target, i, dimension));
    }
    return *target;
}

/**
* @desc: convert napi to float multidim array
* @param env
* @param source
* @param rows
* @param cols
* @param target
* @return target
*/
#define NAPI_TO_FLOAT_MARRAY(env, source, target) NapiToFloatMArray(env, source, target);
#define NAPI_TO_FLOAT_MARRAY_RETVAL(env, source, target, retval) \
  NapiToFloatMArray(env, source, target) \
  return retval;
inline static void NapiToFloatMArray(napi_env env,napi_value* source, float** target){
  uint32_t rows;
  NAPI_CALL(env, napi_get_array_length(env, *source, &rows));

  for(int i=0; i<rows; i++){
    napi_value row;
    NAPI_CALL(env, napi_get_element(env, *source, i, &row));
    uint32_t cols;
    NAPI_CALL(env, napi_get_array_length(env, row, &cols));
    target[i] =  new float[cols];
    
    for(int j=0; j<cols; j++){
      napi_value col;
      NAPI_CALL(env, napi_get_element(env, row, j, &col));
      
      float val;
      NAPI_TO_FLOAT(env, &col, &val);
    
      target[i][j] = 0.0;
    }
  }

  //return target;
}


#endif // CAFFE_NODEJS_UTIL_CONVERSION_H_