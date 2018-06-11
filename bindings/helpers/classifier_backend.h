#ifndef YOLONODEJS_CLASSIFIER_BACKEND_H
#define YOLONODEJS_CLASSIFIER_BACKEND_H

#include <node_api.h>
extern "C" {
    #include "darknet.h"
    #include "matrix.h"
}
#include "../utils/common.h"
#include "../yolo_high_level_api/napi_classifier.h"
using namespace std;

/**
* @description: validate trainer 
* @param classifier: <*Classifier>
*/
/*#define VALIDATE_TRAINER(env, path) ValidateTrainer(env, classifier);
#define VALIDATE_TRAINER_RETVAL(env, classifier, retval) \
  ValidateTrainer(env, classifier); \
  return retval;
static inline void ValidateTrainer(napi_env env, Classifier* classifier){
    bool isDataFile = classifier &&  classifier->dataFilePath_ && classifier->dataFilePath_ != nullptr;
    bool isCfgFile = classifier &&  classifier->cfgFilePath && classifier->cfgFilePath != nullptr;
    bool isResDir = classifier &&  classifier->resultDirPath && classifier->resultDirPath != nullptr;
    bool isGpu = classifier &&  classifier->gpu && classifier->gpu > 0;
    if(!isDataFile){
        NAPI_THROW_ERROR(env, "Data file not found!");
    }
    if(!isCfgFile){
        NAPI_THROW_ERROR(env, "Cfg file not found!");
    }
    if(!isResDir){
        NAPI_THROW_ERROR(env, "Result dir not found!");
    }
    if(!isGpu){
        NAPI_THROW_ERROR(env, "Gpu/Cpu not found!");
    }
}*/


/**
* @description: train images
* @param env: <napi_env>
* @param classifier: <*Classifier>
*/
#define CLASSIFIER_TRAIN(env, path) ClassifierTrain(env, classifier);
#define CLASSIFIER_TRAIN_RETVAL(env, classifier, retval) \
  ClassifierTrain(env, classifier); \
  return retval;
/*static inline void ClassifierTrain(napi_env env, Classifier* classifier){
    int clear = 0;

    int i;
    float avg_loss = -1;
    char *base = basecfg(classifier->cfgFilePath);
    printf("%s\n", base);
    printf("%d\n", classifier->gpu);
    network **nets =(network**) calloc(classifier->gpu, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < classifier->gpu; ++i){
        srand(seed);
        #ifdef GPU
            cuda_set_device(gpus[i]);
        #endif
        nets[i] = load_network(classifier->cfgFilePath, classifier->weightFilePath, clear);
        nets[i]->learning_rate *= classifier->gpu;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * classifier->gpu;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(classifier->dataFilePath);

    
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

            for(i = 0; i < classifier->gpu; ++i){
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
        if(classifier->gpu == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, classifier->gpu, train, 4);
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
            sprintf(buff, "%s/%s_%d.weights",classifier->resultDirPath,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",classifier->resultDirPath,base);
            save_weights(net, buff);
        }
    }

    char buff[256];
    sprintf(buff, "%s/%s.weights", classifier->resultDirPath, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}*/


#endif //YOLONODEJS_CLASSIFIER_BACKEND_H



