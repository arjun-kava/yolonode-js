#ifndef YOLONODEJS_UTILS_H
#define YOLONODEJS_UTILS_H
#include <node_api.h>
extern "C" {
#include "darknet.h"
}
#include "../utils/common.h"

/**
* get current time in timstamp
* @return timestamp
**/
static napi_value yolo_what_time_is_it_now(napi_env env, napi_callback_info info){
    double input = what_time_is_it_now();
    napi_value output;
    NAPI_CALL(env, napi_create_double(env, input, &output));
    return output;
}

#endif //YOLONODEJS_UTILS_H
