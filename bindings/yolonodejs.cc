#include <node_api.h>
#include "utils/common.h"
#include "yolo_low_level_api/binder.cc"
#include "yolo_high_level_api/binder.cc"




/**
* @desc: initialize all factories
**/
napi_value Init(napi_env env, napi_value exports) {
  BIND_LOW_LEVEL_API(env, &exports);
  BIND_HIGH_LEVEL_API(env, &exports);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init);