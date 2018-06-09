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
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 3, NULL);

    int indexM = 0;
    int indexScale = 1;
    IS_OBJECT(env, &args[indexM]);
    IS_NUMBER(env, &args[indexScale]);

    matrix m;
    NAPI_TO_MATRIX(env, &args[indexM],&m)

    float scale;
    NAPI_TO_FLOAT(env,&args[indexScale], &scale);

    scale_matrix(m, scale);

    napi_value vals;
    FLOAT_MARRAY_TO_NAPI(env, m.vals, &m.rows, &m.cols, &vals);

    SET_PROPERTY(env, &args[indexM],"vals", &vals);
    
    return NULL;
}

/**
* @description: resize matrix
* @param m: <matrix>
* @param size: <int>
*/
static napi_value yolo_resize_matrix(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 3, NULL);

    int indexM = 0;
    int indexSize = 1;
    IS_OBJECT(env, &args[indexM]);
    IS_NUMBER(env, &args[indexSize]);

    matrix m;
    NAPI_TO_MATRIX(env, &args[indexM],&m)

    int size;
    NAPI_TO_INT(env,&args[indexSize], &size);

    m = resize_matrix(m, size);

    napi_value vals;
    FLOAT_MARRAY_TO_NAPI(env, m.vals, &m.rows, &m.cols, &vals);

    SET_PROPERTY(env, &args[indexM],"vals", &vals);
    SET_PROPERTY(env, &args[indexM],"rows", &args[indexSize]);

    return args[indexM];
}

/**
* @description: add new matrix
* @param from: <matrix>
* @param to: <matrix>
*/
static napi_value yolo_matrix_add_matrix(napi_env env, napi_callback_info info){
  size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 3, NULL);

    int indexFrom = 0;
    int indexTo = 1;
    IS_OBJECT(env, &args[indexFrom]);
    IS_OBJECT(env, &args[indexTo]);

    matrix from;
    NAPI_TO_MATRIX(env, &args[indexFrom],&from);
    matrix to;
    NAPI_TO_MATRIX(env, &args[indexTo],&to);

    bool isBothMatSame = ( from.rows == to.rows && from.cols == to.cols );
    NAPI_ASSERT(env, isBothMatSame,"Both array needs to have same size!");

    matrix_add_matrix(from, to);

    napi_value vals;
    FLOAT_MARRAY_TO_NAPI(env, from.vals, &from.rows, &from.cols, &vals);

    SET_PROPERTY(env, &args[indexFrom],"vals", &vals);

    return NULL;
}

/**
* @description 
* @param m: <matrix>
*/
static napi_value yolo_copy_matrix(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 2, NULL);

    int indexM = 0;
    IS_OBJECT(env, &args[indexM]);

    matrix m;
    NAPI_TO_MATRIX(env, &args[indexM],&m)

    matrix copy = copy_matrix(m);

    napi_value napi_copy;
    MATRIX_TO_NAPI(env, &copy, &napi_copy);

    return napi_copy;
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

/**
* @description: hold out matrix
* @param m: <matrix>
* @param n: <n>
*/
static napi_value yolo_hold_out_matrix(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 3, NULL);

    int indexM = 0;
    int indexN = 1;
    IS_OBJECT(env, &args[indexM]);
    IS_NUMBER(env, &args[indexN]);
    matrix m;
    NAPI_TO_MATRIX(env, &args[indexM],&m)
    int n;
    NAPI_TO_INT(env,&args[indexN], &n);
    matrix hold = hold_out_matrix(&m, n);
    napi_value napi_hold;
    MATRIX_TO_NAPI(env, &hold, &napi_hold);
    
    return napi_hold;
}

/**
* @description: pop columns
* @param m: <matrix>
* @param c: <int>
* @return col: <float>
*/
static napi_value yolo_pop_column(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 3, NULL);

    int indexM = 0;
    int indexC = 1;
    IS_OBJECT(env, &args[indexM]);
    IS_NUMBER(env, &args[indexC]);

    matrix m;
    NAPI_TO_MATRIX(env, &args[indexM],&m)

    int c;
    NAPI_TO_INT(env,&args[indexC], &c);

    float* col = pop_column(&m, c);

    napi_value napi_col;
    FLOAT_TO_NAPI(env, col, &napi_col);
    
    return napi_col;
}

/**
* @description: convert csv to matrix
* @param path: <string>
*/
static napi_value yolo_csv_to_matrix(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 2, NULL);

    int indexPath = 0;
    IS_STRING(env, &args[indexPath]);

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexPath], &length);
    char* path = new char[length];
    NAPI_TO_CHAR(env,&args[indexPath], path, &length);

    matrix mat = csv_to_matrix(path);

    napi_value napi_mat;
    MATRIX_TO_NAPI(env, &mat, &napi_mat);
    free_matrix(mat);
    return napi_mat;
}

/**
* @description: convert matrix to csv
* @params m: <matrix>
* @params path: <string>
*/
static napi_value yolo_matrix_to_csv(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 2, NULL);

    int indexM = 0;
    IS_OBJECT(env, &args[indexM]);

    matrix m;
    NAPI_TO_MATRIX(env, &args[indexM],&m)

    matrix_to_csv(m);
    free_matrix(m);
    return NULL;
}

/**
* @description: print matrix
* @param m: <matrix>
*/
static napi_value yolo_print_matrix(napi_env env, napi_callback_info info){
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, NULL, NULL));
   
    IS_VALID_NUM_ARG_RETVAL(env, &argc, 2, NULL);

    int indexM = 0;
    IS_OBJECT(env, &args[indexM]);

    matrix m;
    NAPI_TO_MATRIX(env, &args[indexM],&m)

    print_matrix(m);
    free_matrix(m);
    return NULL;
}

#endif //YOLONODEJS_MATRIX_H



