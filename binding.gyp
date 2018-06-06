{
    "variables":{
        'yolo_lib':'/home/arjunkava/Work/yolonode/yolonode-js/node_modules/yolonode-js-build/darknet'
    },
    "targets": [
        {
            "target_name": "yolonodejs",
            "sources": [
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