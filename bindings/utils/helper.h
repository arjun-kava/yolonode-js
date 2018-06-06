#ifndef CAFFE_NODEJS_UTIL_HELPER_H_
#define CAFFE_NODEJS_UTIL_HELPER_H_

#include <node_api.h>

    
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

#endif // CAFFE_NODEJS_UTIL_HELPER_H_

