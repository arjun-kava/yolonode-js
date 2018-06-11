#include <node_api.h>
#include "../utils/common.h"
#include "napi_classifier.h"
//using namespace yolonodejs;
 /**
* @description: factory for classifier
* @param env: <napi_env>
* @param info: <napi_callback_info>
*/
static napi_value CreateClassifier(napi_env env, napi_callback_info info) {
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, nullptr, nullptr, nullptr));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));

    napi_value instance;
    NAPI_CALL(env, Classifier::NewInstance(env, argc, args, &instance));

    return instance;
}

/**
* @description: initialize classifier
* @param env: <napi_env>
*/
static void Initializer(napi_env env){
    Classifier::Init(env);
}


