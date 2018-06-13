const path = require('path');
const { isCPU } = require('./native');
const yoloNodeJsBuild = require( isCPU() ? 'yolonode-js-build' : 'yolonode-js-build-gpu');

// dirs to include
const includeDirs = [
    yoloNodeJsBuild.yoloInclude,
    yoloNodeJsBuild.yoloIncludeSrc,
    !isCPU() ? yoloNodeJsBuild.cudaInclude: "",
    !isCPU() ? yoloNodeJsBuild.cudaTargetInclude: "",
]
// dirs of libraries
const libDirs = [
    yoloNodeJsBuild.yoloLibDir,
    yoloNodeJsBuild.yoloBuildDir,
    !isCPU() ? yoloNodeJsBuild.cudaLib : "",
    !isCPU() ? yoloNodeJsBuild.cudaTargetLib : "",
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
    const yoloModules = yoloNodeJsBuild.libs ? yoloNodeJsBuild.libs: [];
    const libraries = yoloModules.map(lib => console.log(`-l${lib}`));
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

/**
 * @description: getting defines needed
 */
const getDefines = () => {
    const defines = [
        "__cplusplus=1",
        //`GPU=${isCPU()? "0": "1"}`,
        //`LABELS_PATH "${yoloNodeJsBuild.yoloData}/labels"`
    ]
    console.log(defines.join(","));
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
        case '--defines':
            getDefines();
            break;
    }
}