#include <node_api.h>
#include "utils/common.h"
#include "yolo/yolonodejs_image.h"

static napi_value hello(napi_env env, napi_callback_info info) {
  napi_value world;
  const char* str = "world";
  size_t str_len = strlen(str);
  NAPI_CALL(env, napi_create_string_utf8(env, str, str_len, &world));
  return world;
}

/**
* @desc: initialize all factories
**/
napi_value Init(napi_env env, napi_value exports) {
    BindNapiFunction(env, exports, hello, "hello");

    /////// IMAGE
    BindNapiFunction(env, exports, yolo_what_time_is_it_now, "what_time_is_it_now");

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init);