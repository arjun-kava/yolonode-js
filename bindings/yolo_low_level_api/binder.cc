#include <node_api.h>
#include "../utils/common.h"

#include "napi_image.cc"
#include "napi_matrix.cc"
#include "napi_utils.cc"

/**
* @description: bind low level APIs of yolo
* @param exports: <*napi_value>
*/
#define BIND_LOW_LEVEL_API(env, exports) BindLowLevelAPI(env, exports);;
#define BIND_LOW_LEVEL_API_RETVALUE(env, exports, retvalue) \
    BindLowLevelAPI(env, exports) \
    return retvalue;
inline void BindLowLevelAPI(napi_env env,napi_value* exports){
/////// UTILS
  BIND_NAPI_FUN(env, exports, yolo_what_time_is_it_now, "what_time_is_it_now");

  ////// MATRIX
  BIND_NAPI_FUN(env, exports, yolo_matrix_topk_accuracy, "matrix_topk_accuracy");
  BIND_NAPI_FUN(env, exports, yolo_scale_matrix, "scale_matrix");
  BIND_NAPI_FUN(env, exports, yolo_resize_matrix, "resize_matrix");
  BIND_NAPI_FUN(env, exports, yolo_matrix_add_matrix, "matrix_add_matrix");
  BIND_NAPI_FUN(env, exports, yolo_copy_matrix, "copy_matrix");
  BIND_NAPI_FUN(env, exports, yolo_make_matrix, "make_matrix");
  BIND_NAPI_FUN(env, exports, yolo_hold_out_matrix, "hold_out_matrix");
  BIND_NAPI_FUN(env, exports, yolo_pop_column, "pop_column");
  BIND_NAPI_FUN(env, exports, yolo_matrix_to_csv, "matrix_to_csv");
  BIND_NAPI_FUN(env, exports, yolo_csv_to_matrix, "csv_to_matrix");
  BIND_NAPI_FUN(env, exports, yolo_print_matrix, "print_matrix");

  ////// IMAGE
  BIND_NAPI_FUN(env, exports, yolo_get_color, "get_color");

  ///// TEST
  //BIND_NAPI_FUN(env, exports, run, "run");
}
