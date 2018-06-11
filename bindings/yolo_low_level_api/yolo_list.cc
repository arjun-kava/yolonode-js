#ifndef YOLONODEJS_LIST_H
#define YOLONODEJS_LIST_H

#include <node_api.h>
extern "C" {
#include "darknet.h"
#include "list.h"
}
#include "../utils/common.h"
#include "../helpers/common.h"
using namespace std;

/**
* @description: print matrix
* @param m: <matrix>
*/
static napi_value yolo_make_list(napi_env env, napi_callback_info info){
    list* l = make_list();


    return NULL;
}

#endif //YOLONODEJS_LIST_H



