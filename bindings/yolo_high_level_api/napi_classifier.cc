#include "napi_classifier.h"
size_t finalize_count = 0;

Classifier::Classifier() : env_(nullptr), wrapper_(nullptr) {
    this->dataFilePath_ = 0;
    this->cfgFilePath_ = 0;
    this->weightFilePath_ = 0;
    this->gpus_ = GetNumOfThreads() > 0 ? GetNumOfThreads(): 1;
    printf("gpu: %d ",this->gpus_);
    this->gpusList_ = 0;
    this->filePath_ = 0;
    this->labelsPath_ = 0;
    this->trainListPath_ = 0;
    this->testListPath_ = 0;
    this->resultDirPath_ = 0;
    this->outputFilePath_ = 0;
    this->top_ = 2; // predict top 1 class by default
    this->thresh_ = .0;
    this->hierThresh_ = .0;
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
        DECLARE_NAPI_PROPERTY("classify", Classify),
        DECLARE_NAPI_PROPERTY("predict", Predict),
        DECLARE_NAPI_PROPERTY("validate", Validate),
        DECLARE_NAPI_PROPERTY("loadWeights", LoadWeights),

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
        DECLARE_NAPI_GET_SET("resultDirPath", GetResultDirPath, SetResultDirPath),
        DECLARE_NAPI_GET_SET("top", GetTop, SetTop),
        DECLARE_NAPI_GET_SET("thresh", GetThresh, SetThresh),
        DECLARE_NAPI_GET_SET("hierThresh", GetHierThresh, SetHierThresh),
        DECLARE_NAPI_GET_SET("outputFilePath", GetOutputFilePath, SetOutputFilePath)
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
        return nullptr;
    }
    if(!isCfgFile){
        NAPI_THROW_ERROR(env, "CfgFilePath is not set or valid!");
        return nullptr;
    }
    if(!isResDir){
        NAPI_THROW_ERROR(env, "ResultFilePath is not set or valid!");
        return nullptr;
    }
    if(!isGpu){
        NAPI_THROW_ERROR(env, "Gpu/Cpu is not set!");
        return nullptr;
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
            loss = train_networks(nets, classifier->gpus_, train, 4);
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

napi_value Classifier::LoadWeights(napi_env env, napi_callback_info info){
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    //napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, nullptr, &_this, nullptr));  

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    bool isCfgFile = classifier && classifier->cfgFilePath_ &&  classifier->cfgFilePath_ != 0;
    bool isWeightFile = classifier && classifier->weightFilePath_ &&  classifier->weightFilePath_ != 0;
    if(!isCfgFile){
        NAPI_THROW_ERROR(env, "CfgFilePath is not set or valid!");
        return nullptr;
    }
    if(!isWeightFile){
        NAPI_THROW_ERROR(env, "WeightFilePath is not set or valid!");
        return nullptr;
    }

    classifier->net_ = load_network(classifier->cfgFilePath_, classifier->weightFilePath_, 0);
    set_batch_network(classifier->net_, 1);

    return nullptr;
}

napi_value Classifier::Classify(napi_env env,napi_callback_info info){
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    //napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, nullptr, &_this, nullptr));  

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    
    bool isTop = classifier &&  classifier->top_ && classifier->top_ > 0;
    bool isLabelFile = classifier &&  classifier->labelsPath_ && classifier->labelsPath_ != 0;
    bool isfilePath = classifier &&  classifier->filePath_ && classifier->filePath_ != 0;

    
    if(!isTop){
        NAPI_THROW_ERROR(env, "Top is not set or valid!");
        return nullptr;
    }
    if(!isLabelFile){
        NAPI_THROW_ERROR(env, "Label File Path is not set or valid!");
        return nullptr;
    }
    if(!isfilePath){
        NAPI_THROW_ERROR(env, "File Path is not set or valid!");
        return nullptr;
    }

    network *net = classifier->net_;
    set_batch_network(net, 1);
    srand(2222222);
    char **names = get_labels(classifier->labelsPath_);
   
    int i, j;
    const int nsize = 8;
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
  
    if(classifier->filePath_){
        strncpy(input, classifier->filePath_, 256);
    } 
    image im = load_image_color(input,0,0);
    image sized = letterbox_image(im, net->w, net->h);
    layer l = net->layers[net->n-1];


    float *X = sized.data;
    time=what_time_is_it_now();
    network_predict(net, X);
    printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, classifier->thresh_, classifier->hierThresh_, 0, 1, &nboxes);
    printf("nboxes: %d", nboxes);

    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

    napi_value result;
    DetectionToNapi(env,&im, dets, nboxes, names, l.classes, classifier->thresh_, &result);
    free_detections(dets, nboxes);
    
    save_image(im, classifier->outputFilePath_ != 0 ? classifier->outputFilePath_ : "predictions");

    free_image(im);
    free_image(sized);
   
    return result;
}


napi_value Classifier::Predict(napi_env env,napi_callback_info info){
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    //napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, nullptr, &_this, nullptr));  

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    network *net = classifier->net_;
    set_batch_network(net, 1);
    srand(2222222);

    char *name_list = classifier->labelsPath_;
   

    int i = 0;
    char **names = get_labels(name_list);
 
    clock_t time;
    int *indexes =(int*) calloc(classifier->top_, sizeof(int));
    char buff[256];
    char *input = buff;

    strncpy(input, classifier->filePath_, 256);
    
    image im = load_image_color(input, 0, 0);
    image r = letterbox_image(im, net->w, net->h);

    float *X = r.data;
    time=clock();
    float *predictions = network_predict(net, X);
    if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
    top_k(predictions, net->outputs, classifier->top_, indexes);
    fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));

    for(i = 0; i < classifier->top_; ++i){
        int index = indexes[i];
        //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
        //else printf("%s: %f\n",names[index], predictions[index]);
        printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
    }

    if(r.data != im.data) free_image(r);
    free_image(im);

    return nullptr;
}

napi_value Classifier::Validate(napi_env env,napi_callback_info info){
    LOG("in\n");
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    //napi_value args[argc];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, nullptr, &_this, nullptr));  

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    LOG("unwrap\n");
    network *net = classifier->net_;
    
    set_batch_network(net, 1);
    srand(time(0));
    LOG("set batch\n");
    
    char *label_list = classifier->labelsPath_;

    char *valid_list = classifier->testListPath_;
    int classes = 2;
    int topk = 2;

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes =(int*) calloc(topk, sizeof(int));

    for(int i = 0; i < m; ++i){
        int classI = -1;
        char *path = paths[i];
        for(int j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                classI = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == classI) avg_acc += 1;
        for(int j = 0; j < topk; ++j){
            if(indexes[j] == classI) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], classI, pred[0], pred[1]);
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
   
    return nullptr;
}

/************************************
* GETTER & SETTER
*************************************/

napi_value Classifier::GetDataFilePath(napi_env env, napi_callback_info info){
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->dataFilePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetDataFilePath(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->cfgFilePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetCfgFilePath(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->weightFilePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetWeightFilePath(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value gpu;
    INT_TO_NAPI(env, &classifier->gpus_, &gpu);

    return gpu;
}

napi_value Classifier::SetGpu(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->gpusList_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetGPUList(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->filePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetFilePath(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->labelsPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetLabelsPath(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->trainListPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetTrainListPath(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->testListPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetTestListPath(napi_env env, napi_callback_info info){
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
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->resultDirPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetResultDirPath(napi_env env, napi_callback_info info){
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

napi_value Classifier::GetTop(napi_env env, napi_callback_info info){
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value top;
    INT_TO_NAPI(env, &classifier->top_, &top);

    return top;
}

napi_value Classifier::SetTop(napi_env env, napi_callback_info info){
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexTop = 0;
    IS_NUMBER(env, &args[indexTop]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    NAPI_TO_INT(env, &args[indexTop], &classifier->top_);

    return nullptr;
}

napi_value Classifier::GetThresh(napi_env env, napi_callback_info info){
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value thresh;
    FLOAT_TO_NAPI(env, &classifier->thresh_, &thresh);

    return thresh;
}

napi_value Classifier::SetThresh(napi_env env, napi_callback_info info){
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexThresh = 0;
    IS_NUMBER(env, &args[indexThresh]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    NAPI_TO_FLOAT(env, &args[indexThresh], &classifier->thresh_);

    return nullptr;
}

napi_value Classifier::GetHierThresh(napi_env env, napi_callback_info info){
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value hierThresh;
    FLOAT_TO_NAPI(env, &classifier->hierThresh_, &hierThresh);

    return hierThresh;
}

napi_value Classifier::SetHierThresh(napi_env env, napi_callback_info info){
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexHierThresh = 0;
    IS_NUMBER(env, &args[indexHierThresh]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    NAPI_TO_FLOAT(env, &args[indexHierThresh], &classifier->hierThresh_);

    return nullptr;
}

napi_value Classifier::GetOutputFilePath(napi_env env, napi_callback_info info){
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->outputFilePath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetOutputFilePath(napi_env env, napi_callback_info info){
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexOutputFilePath = 0;
    IS_STRING(env, &args[indexOutputFilePath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexOutputFilePath], &length);
    classifier->outputFilePath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexOutputFilePath], classifier->outputFilePath_, &length);

    return nullptr;
}

napi_value Classifier::GetDataDirPath(napi_env env, napi_callback_info info){
    napi_value _this;
    
    NAPI_CALL(env, napi_get_cb_info(env, info, nullptr, nullptr, &_this, nullptr));

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    napi_value dataFilePath;
    CHAR_TO_NAPI(env, classifier->dataDirPath_, &dataFilePath);

    return dataFilePath;
}

napi_value Classifier::SetDataDirPath(napi_env env, napi_callback_info info){
    napi_value _this;
    size_t argc;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, NULL, NULL, NULL));

    napi_value args[argc];
    IS_VALID_NUM_ARG(env, &argc, 2);

    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, &_this, nullptr));  

    int indexDataDirPath = 0;
    IS_STRING(env, &args[indexDataDirPath]);

    Classifier* classifier;
    NAPI_CALL(env, napi_unwrap(env, _this, reinterpret_cast<void**>(&classifier)));

    size_t length;
    GET_NAPI_STRING_LEN(env, &args[indexDataDirPath], &length);
    classifier->dataDirPath_ = new char[length];
    NAPI_TO_CHAR(env,&args[indexDataDirPath], classifier->dataDirPath_, &length);

    EXISTS(env, classifier->dataDirPath_);

    return nullptr;
}


