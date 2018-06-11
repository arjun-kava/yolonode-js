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




#endif //YOLONODEJS_CLASSIFIER_BACKEND_H



