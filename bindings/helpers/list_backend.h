#ifndef YOLONODEJS_MATRIX_BACKEND_H
#define YOLONODEJS_MATRIX_BACKEND_H

#include <node_api.h>
extern "C" {
    #include "darknet.h"
    #include "matrix.h"
}
#include "../utils/common.h"
using namespace std;


/**
* convert matrix to napi object
* @param source: <*list>
* @param target: <*napi_value>
*/
static napi_value ListToNapi(napi_env env,list* source, napi_value* target){

}




#endif //YOLONODEJS_MATRIX_BACKEND_H



