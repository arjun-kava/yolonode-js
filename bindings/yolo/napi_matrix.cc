#ifndef YOLONODEJS_MATRIX_H
#define YOLONODEJS_MATRIX_H

#include <node_api.h>
extern "C" {
#include "darknet.h"
#include "matrix.h"
}
#include "../utils/common.h"
#include "../helpers/common.h"
using namespace std;


/**
* @description: find top k accuracy
* @param truth: <matrix>
* @param guess: <matrix>
* @param k: <int>
*/
static napi_value yolo_matrix_topk_accuracy(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 4, NULL);

    int indexTruth = 0;
    int indexGuess = 1;
    int indexK = 2;
    IS_OBJECT(env, &args[indexTruth]);
    IS_OBJECT(env, &args[indexGuess]);
    IS_NUMBER(env, &args[indexK]);

    matrix truth;
    NAPI_TO_MATRIX(env, &args[indexTruth], &truth);
    
    matrix guess;
    NAPI_TO_MATRIX(env, &args[indexGuess], &guess);
    
    int k;
    NAPI_CALL(env, napi_get_value_int32(env, args[indexK], &k));

    float topk = matrix_topk_accuracy(truth, guess, k);

    free_matrix(truth);
    free_matrix(guess);

    
    napi_value napi_topk;
    FLOAT_TO_NAPI(env, &topk, &napi_topk);

    return napi_topk;
}

/**
* @description: scale matrix
* @param m: <matrix>
* @param scale: <float>
*/
static napi_value yolo_scale_matrix(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 2, NULL);

    int indexM = 0;
    int indexScale = 1;
    IS_OBJECT(env, &args[indexM]);
    IS_NUMBER(env, &args[indexScale]);

    matrix m;
    NAPI_TO_MATRIX(env, &args[indexTruth], &m);

    float scale;
    NAPI_TO_FLOAT(env,&args[indexScale], &scale);

    

    return NULL;
}

/**
* @description: make matrix
* @param rows: <int>
* @param cols: <int>
*/
static napi_value yolo_make_matrix(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 3, NULL);

    int indexRows = 0;
    int indexCols = 1;
    IS_NUMBER(env, &args[indexRows]);
    IS_NUMBER(env, &args[indexCols]);

    int32_t rows = -1;
    int32_t cols = -1;
    NAPI_CALL(env, napi_get_value_int32(env, args[indexRows], &rows));
    NAPI_CALL(env, napi_get_value_int32(env, args[indexCols], &cols));

    napi_value result;
    CREATE_MATRIX_RETVAL(env, &rows, &cols, &result, NULL);

    return  result;
}

#endif //YOLONODEJS_MATRIX_H



