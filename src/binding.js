const path = require('path');
const { isCPU } = require('./native');
const yoloNodeJsBuild = require( isCPU() ? 'yolonode-js-build' : 'yolonode-js-build-gpu');
const cvNodeJsBuild = require( isCPU() ? 'opencvnode-js-build' : 'opencvnode-js-build-gpu');

// dirs to include
const includeDirs = [
    yoloNodeJsBuild.yoloInclude,
    yoloNodeJsBuild.yoloIncludeSrc,
    cvNodeJsBuild.opencvInclude,
    cvNodeJsBuild.opencvIncludeCC,
]
// dirs of libraries
const libDirs = [
    yoloNodeJsBuild.yoloLibDir,
    cvNodeJsBuild.opencvLibDir,
    yoloNodeJsBuild.yoloBuildDir
]

/**
 * @description: fetching needed include directories for binding
 */
const getIncludeDirs = () => {
    console.log(includeDirs.join(" "));
}

/**
 * @description: fetching needed libraries for bindings
 */
const getLibraries = () => {
    const common = ["m",
    "pthread",
    "stdc++"]
    const yoloModules = yoloNodeJsBuild.yoloModules ? yoloNodeJsBuild.yoloModules: [];
    const cvModules = cvNodeJsBuild.libs ? cvNodeJsBuild.libs: [];
    let libraries = (common).concat(yoloModules)//.concat(cvModules);
    libraries = libraries.map(lib => console.log(`-l${lib}`));
    console.log(libraries.join(" "))
}

/**
 * @description: fetching needed lib directories
 */
const getLibDirs = () => {
    console.log(libDirs.join(" "))
}

/**
 * @description: fetching needed cf flags 
 */
const getCFlags = () => {
    const flags =  libDirs.map(dir => `-Wl,-rpath,${dir}`);
    console.log(flags.join(" "));
}

const bindingArgIndex = 2;
const isBindingArg = process.argv && process.argv.length == (bindingArgIndex + 1) && process.argv[bindingArgIndex];
if(isBindingArg){
    switch(process.argv[bindingArgIndex]){
        case '--include_dirs':
            getIncludeDirs();
            break;
        case '--libraries':
            getLibraries();
            break;
        case '--library_dirs':
            getLibDirs();
            break;
        case '--ldflags':
            getCFlags();
            break;
    }
}