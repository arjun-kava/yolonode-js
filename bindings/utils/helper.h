#ifndef CAFFE_NODEJS_UTIL_HELPER_H_
#define CAFFE_NODEJS_UTIL_HELPER_H_

#include <node_api.h>

#define ARRAY_SIZE(array) (sizeof(array) / sizeof(array[0]))

/**
* @desc: bind function to export container
* @param env
* @param exports
* @param fn
* @param name
**/
inline bool BindNapiFunction(napi_env env,napi_value exports,napi_callback fn,const char* name){
    napi_value fnRef;
    NAPI_CALL(env, napi_create_function(env, NULL, NAPI_AUTO_LENGTH, fn, NULL, &fnRef));
    NAPI_CALL(env, napi_set_named_property(env, exports, name, fnRef));
    return true;
}

/**
* convert float array to js array
* @param env: <napi_env> 
* @param source: <*float> 
* @param target: <*napi_value> 
*/
#define FLOAT_ARRAY_TO_NAPI(env, source, target) \
  if (!FloatArrayToNapi(env, source, target)) return NULL;
#define FLOAT_ARRAY_TO_NAPI_RETVAL(env, source, target, retval) \
  if (!FloatArrayToNapi(env, source, target)) return retval;
static napi_value FloatArrayToNapi(napi_env env,float* source, napi_value* target){
    NAPI_CALL(env, napi_create_array(env, target));
    for(int i = 0; i < sizeof(*source); i++)
    {
      napi_value elem;
      NAPI_CALL(env, napi_create_int64(env, (int64_t) *(source + i), &elem));
      NAPI_CALL(env, napi_set_element(env, *target, i, elem));
    }
    return *target;
}

#endif // CAFFE_NODEJS_UTIL_HELPER_H_

