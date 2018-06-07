#ifndef YOLONODEJS_IMAGE_H
#define YOLONODEJS_IMAGE_H
#include <node_api.h>
#include <sys/time.h>
extern "C" {
#include "darknet.h"
}
#include "../utils/common.h"

//static napi_value yolo_get_color(napi_env env, napi_callback_info info)
static napi_value yolo_what_time_is_it_now(napi_env env, napi_callback_info info){
    double input = what_time_is_it_now();
    napi_value output;
    NAPI_CALL(env, napi_create_double(env, input, &output));
    return output;
}

#endif //YOLONODEJS_IMAGE_H
