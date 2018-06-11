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

/**
* @description: bind this from request
* @param env: <napi_env>
* @param info: <napi_callbacl_info>
* @param _this: <napi_value>
*/
#define BIND_THIS(env, info, _this, argc, args) BindThis(env, info, _this, argc, args);
#define BIND_THIS_RETVAL(env, info, _this, argc, args, retval) \
  BindThis(env, info, _this, argc, args) \
  return retval;
inline static napi_value BindThis(napi_env env,napi_callback_info* info,
   napi_value* _this, size_t* argc = nullptr, napi_value* args = nullptr){
   //NAPI_CALL(env,
   //   napi_get_cb_info(env, *info, argc, *args, _this, nullptr));
      return *_this;
}


#endif //CAFFE_NODEJS_UTIL_OBJECT_H_

