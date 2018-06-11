#include "napi_classifier.h"
size_t finalize_count = 0;

Classifier::Classifier() : env_(nullptr), wrapper_(nullptr) {
    this->dataFilePath_ = 0;
    this->cfgFilePath_ = 0;
    this->weightFilePath_ = 0;
    this->gpus_ = GetNumOfThreads() > 0 ? GetNumOfThreads(): 1;
    this->gpusList_ = 0;
    this->filePath_ = 0;
    this->labelsPath_ = 0;
    this->trainListPath_ = 0;
    this->testListPath_ = 0;
    this->resultDirPath_ = 0;
}

Classifier::~Classifier() {
    finalize_count++;
    delete dataFilePath_;
    delete cfgFilePath_;
    delete weightFilePath_;
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

        DECLARE_NAPI_PROPERTY("train", Train),

        // GETTER / SETTER
        DECLARE_NAPI_GET_SET("dataFilePath", GetDataFilePath, SetDataFilePath),
        DECLARE_NAPI_GET_SET("cfgFilePath", GetCfgFilePath, SetCfgFilePath),
        DECLARE_NAPI_GET_SET("weightFilePath", GetWeightFilePath, SetWeightFilePath),
        DECLARE_NAPI_GET_SET("gpu", GetGpu, SetGpu),
        DECLARE_NAPI_GET_SET("gpusList", GetGPUList, SetGPUList),
        DECLARE_NAPI_GET_SET("filePath", GetFilePath, SetFilePath),
        DECLARE_NAPI_GET_SET("labelsPath", GetLabelsPath, SetLabelsPath),
        DECLARE_NAPI_GET_SET("trainListPath", GetTrainListPath, SetTrainListPath),
        DECLARE_NAPI_GET_SET("testListPath", GetTestListPath, SetTestListPath),
        DECLARE_NAPI_GET_SET("resultDirPath", GetResultDirPath, SetResultDirPath)
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

napi_value Classifier::Train(napi_env env, napi_callback_info info){
    LOG("calling Classifier::train()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    //napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, nullptr, &_this, nullptr));  

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));


    bool isDataFile = classifier &&  classifier->dataFilePath_ && classifier->dataFilePath_ != nullptr;
    bool isCfgFile = classifier &&  classifier->cfgFilePath_ && classifier->cfgFilePath_ != nullptr;
    bool isResDir = classifier &&  classifier->resultDirPath_ && classifier->resultDirPath_ != nullptr;
    bool isGpu = classifier &&  classifier->gpus_ && classifier->gpus_ > 0;
    if(!isDataFile){
        NAPI_THROW_ERROR(env, "DataFilePath is not set or valid!");
    }
    if(!isCfgFile){
        NAPI_THROW_ERROR(env, "CfgFilePath is not set or valid!");
    }
    if(!isResDir){
        NAPI_THROW_ERROR(env, "ResultFilePath is not set or valid!");
    }
    if(!isGpu){
        NAPI_THROW_ERROR(env, "Gpu/Cpu is not set!");
    }

      int clear = 0;

    int i;
    float avg_loss = -1;
    char *base = basecfg(classifier->cfgFilePath_);
    printf("%s\n", base);
    printf("%d\n", classifier->gpus_);
    network **nets =(network**) calloc(classifier->gpus_, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < classifier->gpus_; ++i){
        srand(seed);
        #ifdef GPU
            cuda_set_device(gpus[i]);
        #endif
        nets[i] = load_network(classifier->cfgFilePath_, classifier->weightFilePath_, clear);
        nets[i]->learning_rate *= classifier->gpus_;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * classifier->gpus_;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(classifier->dataFilePath_);

    
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;
    int epoch = (*net->seen)/N;

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(net->random && count++%40 == 0){
            printf("Resizing\n");
            int dim = (rand() % 11 + 4) * 32;
            //if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;
            args.size = dim;
            args.min = net->min_ratio*dim;
            args.max = net->max_ratio*dim;
            printf("%d %d\n", args.min, args.max);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < classifier->gpus_; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(classifier->gpus_ == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, classifier->gpu_, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",classifier->resultDirPath_,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",classifier->resultDirPath_,base);
            save_weights(net, buff);
        }
    }

    char buff[256];
    sprintf(buff, "%s/%s.weights", classifier->resultDirPath_, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);

    return nullptr;
}


/************************************
* GETTER & SETTER
*************************************/

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

    EXISTS(env, classifier->dataFilePath_);

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

    EXISTS(env, classifier->cfgFilePath_);

    return nullptr;
}

napi_value Classifier::GetWeightFilePath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetWeightFilePath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->weightFilePath_, &dataFilePath);

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
    classifier->weightFilePath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexDataWeightPath], classifier->weightFilePath_, &length);

    EXISTS(env, classifier->weightFilePath_);

    return nullptr;
}

napi_value Classifier::GetGpu(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetGpu()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value gpu;
    INT_TO_NAPI(env, &classifier->gpus_, &gpu);

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

    NAPI_TO_INT(env, &args[indexGpu], &classifier->gpus_)

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

    EXISTS(env, classifier->filePath_);

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

    EXISTS(env, classifier->labelsPath_);

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

    EXISTS(env, classifier->trainListPath_);

    return nullptr;
}

napi_value Classifier::GetTestListPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetTestListPath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->testListPath_, &dataFilePath);

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

    int indexTestPath = 0;
    IS_STRING(env, &args[indexTestPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexTestPath], &length);
    classifier->testListPath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexTestPath], classifier->testListPath_, &length);

    EXISTS(env, classifier->testListPath_);

    return nullptr;
}

napi_value Classifier::GetResultDirPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::GetResultDirPath()");
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->resultDirPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetResultDirPath(napi_env env, napi_callback_info info){
    LOG("calling Classifier::SetResultDirPath()");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexResultDirPath = 0;
    IS_STRING(env, &args[indexResultDirPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexResultDirPath], &length);
    classifier->resultDirPath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexResultDirPath], classifier->resultDirPath_, &length);

    EXISTS(env, classifier->resultDirPath_);


    return nullptr;
}



