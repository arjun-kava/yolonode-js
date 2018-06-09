#ifndef CAFFE_NODEJS_UTIL_VALIDATE_H_
#define CAFFE_NODEJS_UTIL_VALIDATE_H_

#include <node_api.h>
#include "debug.h"


#define IS_CONSTRUCTOR(env, info) \
  if (!IsConstructor(env, info, __FILE__, __LINE__)) return true;

#define IS_CONSTRUCTOR_RETVAL(env, info, retval) \
  if (!IsConstructor(env, info, __FILE__, __LINE__)) return retval;

/**
 * @desc: validate valid call of constructor
 **/
inline bool IsConstructor(napi_env env, napi_callback_info info,
                          const char* file, const size_t lineNumber) {
  napi_value target;
  NAPI_CALL(env, napi_get_new_target(env, info, &target));
  bool is_target = target != NULL;
  if (!is_target) {
      NAPI_THROW_ERROR(env, "Invalid Constructor Call!")
  }
  return is_target;
}

#define IS_VALID_NUM_ARG(env, argc, numOfArg) \
  if (!IsValidNumberOfArg(env, argc, numOfArg, __FILE__, __LINE__)) return true;

#define IS_VALID_NUM_ARG_RETVAL(env, argc, numOfArg, retval) \
  if (!IsValidNumberOfArg(env, argc, numOfArg, __FILE__, __LINE__)) return retval;

/**
* @desc: check valid number of argument
* @param argc: <size_t>
* @params numOfArg
**/
inline bool IsValidNumberOfArg(napi_env env, size_t* argc, int numOfArg,
                                  const char* file, const size_t lineNumber){
  if(*argc == 0 || *argc >= numOfArg){
    NAPI_THROW_ERROR(env, "Invalid Number of arguments!");
    return false;
  }
  else 
  {
    return true;
  }
}


/**
 * @description: validate valid number
 * @param env
 * @param value
 ***/
 #define IS_NUMBER(env, value) IsNum(env, value);
#define IS_NUMBER_RETVALUE(env, value, retvalue) \
    IsNum(env, value); \
    return retvalue;
inline bool IsNum(napi_env env, napi_value* value){
  napi_valuetype type;
  if(*value){
    NAPI_CALL(env, napi_typeof(env, *value, &type));
    bool is_number = type == napi_number;
    if(!is_number){
      NAPI_THROW_ERROR(env, "Invalid Number!");
      return false;
    }
    return is_number;
  }
  else{
      NAPI_THROW_ERROR(env, "Invalid Number!");
      return false;
  }
}

/**
 * @description: validate valid string path
 * @param env
 * @param value
 ***/
#define IS_STRING(env, value) IsString(env, value);
#define IS_STRING_RETVALUE(env, value, retvalue) \
    IsString(env, value); \
    return retvalue;
inline bool IsString(napi_env env, napi_value* value){
  napi_valuetype type;
  if(*value){
    NAPI_CALL(env, napi_typeof(env, *value, &type));
    bool is_string = type == napi_string;
    if(!is_string){
      NAPI_THROW_ERROR(env, "Invalid String!");
      return false;
    }
    return is_string;
  }
  else{
      NAPI_THROW_ERROR(env, "Invalid String!");
      return false;
  }
}

/**
* check whether argument is array
* @param env: <napi_env>
* @param array: <napi_value>
*/
#define IS_ARRAY(env, array) \
    if(!IsArray(env, array)) return;
#define IS_ARRAY_RETVALUE(env, array, retvalue) \
    if(!IsArray(env, array)) return retvalue;
inline bool IsArray(napi_env env, napi_value array){
  napi_valuetype type;
  NAPI_CALL(env, napi_typeof(env, array, &type));
  if(type != napi_object){
    NAPI_THROW_ERROR(env, "Wrong type of arguments. Expect Array!");
    return false;
  }
  return true;
}

/**
* @description: validate argument as object
* @parma env: <napi_env>
* @parma object: <napi_value>
*/
#define IS_OBJECT(env, object) IsObject(env, object);;
#define IS_OBJECT_RETVALUE(env, object, retvalue) \
    IsObject(env, object) \
    return retvalue;
inline bool IsObject(napi_env env, napi_value* object){
  napi_valuetype type;
  NAPI_CALL(env, napi_typeof(env, *object, &type));
  if(type != napi_object){
    NAPI_THROW_ERROR(env, "Wrong type of arguments. Expect Object!");
    return false;
  }
  return true;
}


#endif  // CAFFE_NODEJS_UTIL_VALIDATE_H_