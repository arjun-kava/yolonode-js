#include "yolonodejs_image.h"


static napi_value yolo_get_color(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));

    IS_VALID_NUM_RETVAL(env, &argc, 3, NULL);

    int indexC = 0;
    int indexX = 1;
    int indexMax = 2;
    IS_NUMBER_RETVALUE(env, args[indexC], NULL);
    IS_NUMBER_RETVALUE(env, args[indexX], NULL);
    IS_NUMBER_RETVALUE(env, args[indexMax], NULL);

    int32_t c = -1;
    int32_t x = -1;
    int32_t max = -1;
    NAPI_CALL(env, napi_get_value_int32(env, args[indexC], &c));
    NAPI_CALL(env, napi_get_value_int32(env, args[indexX], &x));
    NAPI_CALL(env, napi_get_value_int32(env, args[indexMax], &max));

    float r = get_color(c,x,(float)max);
    napi_value result = NULL;
    ConvertIntToNapi(env, &r, &result);

    return  result;
}