{
    "variables":{
        'yolo_lib':'/home/arjunkava/Work/yolonode/yolonode-js/node_modules/yolonode-js-build/darknet'
    },
    "targets": [
        {
            "defines":[
                "__cplusplus=1"
            ],
            "target_name": "yolonodejs",
            "sources": [
                #"bindings/yolo/yolonodejs_image.cc",
                "bindings/yolo/yolonodejs_image.h",
                "bindings/yolonodejs.cc",
            ],
            'include_dirs': [
                "<!@(node ./src/binding.js --include_dirs)"
            ],
            'library_dirs' : [
                "<!@(node ./src/binding.js --library_dirs)"
            ],
            'libraries': [
                "<!@(node ./src/binding.js --libraries)"
            ],
            'ldflags': [
                "<!@(node ./src/binding.js --ldflags)"
            ],
            "cflags" : [
    			"-std=c++11",
                "-Wall",
                "-Wfatal-errors",
                "-fPIC",
                "-Ofast"
            ],
            "cflags!" : [
                "-fno-exceptions",
                "-fno-conversion-null"  
            ],
            "cflags_cc!": [
                "-fno-rtti",
                "-fno-exceptions",
                "-Wno-ignored-qualifiers"            
            ],
            "configurations": {
                "Debug": {
                    "cflags": ["--coverage"],
                    "ldflags": ["--coverage"]
                },
            }
        }
    ]
}