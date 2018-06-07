const packageJson = require('../package.json')
/**
 * check whether user have defined to build on basis of GPU
 * default: CPU MODE
 */
const isCPU_ = () => {
    const isPackage = packageJson && packageJson.name;
    if(isPackage && packageJson.name == 'yolonode-js'){
        return 1
    }
    else if(isPackage && packageJson.name == 'yolonode-js-gpu') {
        return 0;
    }
    else{
        return 1;
    }
}

module.exports = {
    isCPU: isCPU_
}