#include <node_api.h>
#include "utils/common.h"

/**
* @desc: initialize all factories
**/
napi_value Init(napi_env env, napi_value exports) {

    napi_property_descriptor desc[] = {};

    napi_define_properties(env, exports, ARRAY_SIZE(desc), desc);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init);