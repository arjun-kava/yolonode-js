#ifndef YOLO_NODEJS_CLASSIFIER_H_
#define YOLO_NODEJS_CLASSIFIER_H_

#include <node_api.h>
extern "C" {
#include "darknet.h"
#include "matrix.h"
}
#include "../utils/common.h"
#include "../helpers/common.h"
using namespace std;

class Classifier{
    public:
        static napi_status Init(napi_env env);
        static void Destructor(napi_env env, void* nativeObject, void* finalize_hint);
        static napi_status NewInstance(napi_env env, size_t argc, napi_value args[],
                                        napi_value* instance);

    private:
        explicit Classifier();
        static napi_ref constructor;
        static napi_value New(napi_env env, napi_callback_info info);
        ~Classifier();

        static napi_value GetDataFilePath(napi_env env, napi_callback_info info);
        static napi_value SetDataFilePath(napi_env env, napi_callback_info info);

        static napi_value GetCfgFilePath(napi_env env, napi_callback_info info);
        static napi_value SetCfgFilePath(napi_env env, napi_callback_info info);

        static napi_value GetWeightFilePath(napi_env env, napi_callback_info info);
        static napi_value SetWeightFilePath(napi_env env, napi_callback_info info);

        static napi_value GetGpu(napi_env env, napi_callback_info info);
        static napi_value SetGpu(napi_env env, napi_callback_info info);

        static napi_value GetGPUList(napi_env env, napi_callback_info info);
        static napi_value SetGPUList(napi_env env, napi_callback_info info);

        static napi_value GetFilePath(napi_env env, napi_callback_info info);
        static napi_value SetFilePath(napi_env env, napi_callback_info info);

        static napi_value GetLabelsPath(napi_env env, napi_callback_info info);
        static napi_value SetLabelsPath(napi_env env, napi_callback_info info);

        static napi_value GetTrainListPath(napi_env env, napi_callback_info info);
        static napi_value SetTrainListPath(napi_env env, napi_callback_info info);

         static napi_value GetTestListPath(napi_env env, napi_callback_info info);
        static napi_value SetTestListPath(napi_env env, napi_callback_info info);


        napi_env env_;
        napi_ref wrapper_;

        char* dataFilePath_;
        char* cfgFilePath_;
        char* weighFilePath_;
        int* gpus_;
        char* gpusList_;
        char* filePath_;
        char* labelsPath_;
        char* trainListPath_;
        char* testListPath_;
};



#endif //YOLO_NODEJS_CLASSIFIER_H_