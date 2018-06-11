#include <node_api.h>
#include "../utils/common.h"
#include "napi_factory.cc"

/**
* @description: bind low level APIs of yolo
* @param exports: <*napi_value>
*/
#define BIND_HIGH_LEVEL_API(env, exports) BindHighLevelAPI(env, exports);
#define BIND_HIGH_LEVEL_API_RETVALUE(env, exports, retvalue) \
    BindHighLevelAPI(env, exports) \
    return retvalue;
inline void BindHighLevelAPI(napi_env env,napi_value* exports){
    // initialize all class 
    Initializer(env);

    napi_property_descriptor desc[] = {
        DECLARE_NAPI_PROPERTY("classifier", CreateClassifier)
    };

    napi_define_properties(env, *exports, ARRAY_SIZE(desc), desc);
}



