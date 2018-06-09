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
* @param mat: <*matrix>
* @param object: <*napi_value>
*/
#define MATRIX_TO_NAPI(env, mat, object) \
  if (!MatrixToNapi(env, mat, object)) return NULL;
#define MATRIX_TO_NAPI_RETVAL(env, mat, object, retval) \
  if (!MatrixToNapi(env, mat, object)) return retval;
static napi_value MatrixToNapi(napi_env env, matrix* mat, napi_value* object){
    NAPI_CALL(env, napi_create_object(env, object));
    
    napi_value rows;
    napi_value cols;
    int int_rows = mat->rows;
    int int_cols = mat->cols;
    float **float_vals = mat->vals;
    INT_TO_NAPI_RETVAL(env, &int_rows, &rows, NULL);
    INT_TO_NAPI_RETVAL(env, &int_cols, &cols, NULL);

    napi_value vals;
    FLOAT_MARRAY_TO_NAPI(env, float_vals, &int_rows, &int_cols, &vals);

    NAPI_CALL(env, napi_set_named_property(env, *object, "rows", rows));
    NAPI_CALL(env, napi_set_named_property(env, *object, "cols", cols));
    NAPI_CALL(env, napi_set_named_property(env, *object, "vals", vals));
    return  *object;
}

/**
* @description: convert napi matrix to mat
* @param object: <*napi_value>
* @param mat: <*matrix>
*/
#define NAPI_TO_MATRIX(env, object, mat) NapiToMatrix(env, object, mat);
#define NAPI_TO_MATRIX_RETVAL(env, object, mat, retval) \
  if (!NapiToMatrix(env, object, mat)) return retval;
static matrix NapiToMatrix(napi_env env, napi_value* object, matrix* mat){
  

  napi_value napi_rows;
  GET_PROPERTY(env, object, "rows", &napi_rows);

  napi_value napi_cols;
  GET_PROPERTY(env, object, "cols", &napi_cols);

  napi_value napi_vals;
  GET_PROPERTY(env, object, "vals", &napi_vals);

  uint32_t i, length;
  NAPI_CALL(env, napi_get_array_length(env, napi_vals, &length));

  NAPI_TO_INT(env, &napi_rows, &mat->rows);
  NAPI_TO_INT(env, &napi_cols, &mat->cols);

  mat->vals = new float*[mat->rows];
  NAPI_TO_FLOAT_MARRAY(env, &napi_vals, mat->vals);

  return *mat;
}

/**
* @description: create matrix 
* @param rows: <*int>
* @param cows: <*int>
* @param result: <*napi_value>
*/
#define CREATE_MATRIX(env, rows, cols, result) \
  if (!MatrixToObject(env, rows, cols, result)) return NULL;
#define CREATE_MATRIX_RETVAL(env, rows, cols, result, retval) \
  if (!CreateMatrix(env, rows, cols, result)) return retval;
static napi_value CreateMatrix(napi_env env, int* rows, int* cols, napi_value* result){
  matrix mat = make_matrix(*rows, *cols);
  MATRIX_TO_NAPI(env, &mat, result);
  free_matrix(mat);
  return *result;
}




#endif //YOLONODEJS_MATRIX_BACKEND_H



