#include <node_api.h>
#include "utils/common.h"
#include "yolo/napi_image.cc"
#include "yolo/napi_matrix.cc"
#include "yolo/napi_utils.cc"


/**
* @desc: initialize all factories
**/
napi_value Init(napi_env env, napi_value exports) {

  /////// UTILS
  BIND_NAPI_FUN(env, exports, yolo_what_time_is_it_now, "what_time_is_it_now");

  ////// MATRIX
  BIND_NAPI_FUN(env, exports, yolo_matrix_topk_accuracy, "matrix_topk_accuracy");
  BIND_NAPI_FUN(env, exports, yolo_make_matrix, "make_matrix");

  ////// IMAGE
  BIND_NAPI_FUN(env, exports, yolo_get_color, "get_color");

  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init);