#include "napi_classifier.h"

size_t finalize_count = 0;

Classifier::Classifier() : env_(nullptr), wrapper_(nullptr) {
    gpus_ = 1;
}

Classifier::~Classifier() {
    finalize_count++;
    delete dataFilePath_;
    delete cfgFilePath_;
    delete weighFilePath_;
    delete gpus_;
    delete gpusList_;
    delete filePath_;
    delete labelsPath_;
    delete trainListPath_;
    delete testListPath_;
    napi_delete_reference(env_, wrapper_);
}

void Classifier::Destructor(napi_env env, void *nativeObject,
                            void * /*finalize_hint*/) {
    Classifier *classify = static_cast<Classifier *>(nativeObject);
    delete classify;
}
napi_ref Classifier::constructor;

napi_status Classifier::Init(napi_env env) {
    napi_status status;
    napi_property_descriptor properties[] = {
        DECLARE_NAPI_GET_SET("dataFilePath", GetDataFilePath, SetDataFilePath),
        DECLARE_NAPI_GET_SET("cfgFilePath", GetCfgFilePath, SetCfgFilePath),
        DECLARE_NAPI_GET_SET("weighFilePath", GetWeightFilePath, SetWeightFilePath),
        DECLARE_NAPI_GET_SET("gpu", GetGpu, SetGpu),
        DECLARE_NAPI_GET_SET("gpusList", GetGPUList, SetGPUList),
        DECLARE_NAPI_GET_SET("filePath", GetFilePath, SetFilePath),
        DECLARE_NAPI_GET_SET("labelsPath", GetLabelsPath, SetLabelsPath),
        DECLARE_NAPI_GET_SET("trainListPath", GetTrainListPath, SetTrainListPath),
        DECLARE_NAPI_GET_SET("testListPath", GetTestListPath, SetTestListPath)
    };

    napi_value cons;
    NAPI_CALL(env,napi_define_class(env, "Classifier", -1, New, nullptr, ARRAY_SIZE(properties), properties, &cons));
    NAPI_CALL(env, napi_create_reference(env, cons, 1, &constructor));
    return napi_ok;
}

napi_value Classifier::New(napi_env env, napi_callback_info info) {
    IS_CONSTRUCTOR(env, &info);
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    napi_value _this;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, NULL));

    Classifier* classify = new Classifier();

    classify->env_ = env;
    NAPI_CALL(env, napi_wrap(env, 
                        _this, 
                        classify, 
                        Classifier::Destructor,
                        nullptr,  // finalize_hint
                        &classify->wrapper_));
}

napi_status Classifier::NewInstance(napi_env env, size_t argc, napi_value args[],
                                    napi_value *instance) {
    napi_status status;
    napi_value cons;
    status = napi_get_reference_value(env, constructor, &cons);
    NAPI_ASSERT(env, (status == napi_ok),"Failed to initialize construtor!");
    status = napi_new_instance(env, cons, argc, args, instance);
    NAPI_ASSERT(env, (status == napi_ok),"Failed to create instance!");
    return napi_ok;
}

napi_value Classifier::GetDataFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetDataFilePath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->dataFilePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetDataFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetDataFilePath()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexDataFilePath = 0;
    IS_STRING(env, &args[indexDataFilePath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexDataFilePath], &length);
    classifier->dataFilePath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexDataFilePath], classifier->dataFilePath_, &length);

    return nullptr;
}

napi_value Classifier::GetCfgFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetCfgFilePath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->cfgFilePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetCfgFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetCfgFilePath()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexDataCfgPath = 0;
    IS_STRING(env, &args[indexDataCfgPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexDataCfgPath], &length);
    classifier->cfgFilePath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexDataCfgPath], classifier->cfgFilePath_, &length);

    return nullptr;
}

napi_value Classifier::GetWeightFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetWeightFilePath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->weighFilePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetWeightFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetWeightFilePath()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexDataWeightPath = 0;
    IS_STRING(env, &args[indexDataWeightPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexDataWeightPath], &length);
    classifier->weighFilePath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexDataWeightPath], classifier->weighFilePath_, &length);

    return nullptr;
}

napi_value Classifier::GetGpu(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetGpu()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value gpu;
    INT_TO_NAPI(env, classifier->gpus_, &gpu);

    return gpu;
}

napi_value Classifier::SetGpu(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetGpu()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexGpu = 0;
    IS_NUMBER(env, &args[indexGpu]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    NAPI_TO_INT(env, &args[indexGpu], classifier->gpus_)

    return nullptr;
}

napi_value Classifier::GetGPUList(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetGPUList()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->gpusList_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetGPUList(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetGPUList()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexDataWeightPath = 0;
    IS_STRING(env, &args[indexDataWeightPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexDataWeightPath], &length);
    classifier->gpusList_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexDataWeightPath], classifier->gpusList_, &length);

    return nullptr;
}

napi_value Classifier::GetFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetFilePath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->filePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetGPUList()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexFilePath = 0;
    IS_STRING(env, &args[indexFilePath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexFilePath], &length);
    classifier->filePath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexFilePath], classifier->filePath_, &length);

    return nullptr;
}

napi_value Classifier::GetLabelsPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetLabelsPath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->labelsPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetLabelsPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetLabelsPath()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexLabelsPath = 0;
    IS_STRING(env, &args[indexLabelsPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexLabelsPath], &length);
    classifier->labelsPath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexLabelsPath], classifier->labelsPath_, &length);

    return nullptr;
}

napi_value Classifier::GetTrainListPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetTrainListPath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->trainListPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetTrainListPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetTrainListPath()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexLabelsPath = 0;
    IS_STRING(env, &args[indexLabelsPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexLabelsPath], &length);
    classifier->trainListPath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexLabelsPath], classifier->trainListPath_, &length);

    return nullptr;
}

napi_value Classifier::GetTestListPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetTestListPath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->trainListPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetTestListPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetTestListPath()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexLabelsPath = 0;
    IS_STRING(env, &args[indexLabelsPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexLabelsPath], &length);
    classifier->trainListPath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexLabelsPath], classifier->trainListPath_, &length);

    return nullptr;
}



