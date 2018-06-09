#ifndef CAFFE_NODEJS_UTIL_OBJECT_H_
#define CAFFE_NODEJS_UTIL_OBJECT_H_

#include <node_api.h>

/**
* check whether object have property
* @param env: <napi_env>
* @param object: <*napi_value>
* @param key: <*napi_value>
*/
#define HAS_PROPERTY(env, object, key) HasProperty(env, object, key);
#define HAS_PROPERTY_RETVAL(env, object, key, retval) \
  if (HasProperty(env, object, key)) return retval;
inline static bool HasProperty(napi_env env,napi_value* object, napi_value* key){
  bool has_property;
  NAPI_CALL(env, napi_has_property(env, *object, *key, &has_property));
  return has_property;
}

/**
* convert float array to js array
* @param env: <napi_env> 
* @param source: <*float> 
* @param target: <*napi_value> 
*/
#define GET_PROPERTY(env, object, key, result) GetProperty(env, object, key, result);
#define GET_PROPERTY_RETVAL(env, object, key, result, retval) \
  if (GetProperty(env, object, key, result)) return retval;
 static napi_value GetProperty(napi_env env, napi_value* object,
                                          const char* key, napi_value* result){
    napi_value target_key;
    CHAR_TO_NAPI(env, key, &target_key);
    bool has_property = HAS_PROPERTY(env, object, &target_key);
    if(has_property){
      NAPI_CALL(env, napi_get_property(env, *object, target_key, result));
      return *result;
    }
    else 
    {
      NAPI_THROW_ERROR(env, "Accessing Invalid Property!");
      return *result;
    }
}

#define SET_PROPERTY(env, object, key, result) SetProperty(env, object, key, result);
#define SET_PROPERTY_RETVAL(env, object, key, result, retval) \
  SetProperty(env, object, key, result) \
  return retval;
static napi_value SetProperty(napi_env env, napi_value* object,const char* key, napi_value* result){
    napi_value target_key;
    CHAR_TO_NAPI(env, key, &target_key);
    bool has_property = HAS_PROPERTY(env, object, &target_key);
    if(has_property){
      NAPI_CALL(env, napi_set_property(env, *object, target_key, *result));
      return *result;
    }
    else 
    {
      NAPI_THROW_ERROR(env, "Accessing Invalid Property!");
      return *result;
    }
}




#endif // CAFFE_NODEJS_UTIL_OBJECT_H_

