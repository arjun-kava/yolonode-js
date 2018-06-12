var fs = require('fs');
var path = require('path');
const randomInt = () => {
    return parseInt(Math.random() * 100);
}

/**
 * @description: write tiny data file
 * @param {string} classes
 * @param {string} trainFilePath 
 * @param {string} labelFilePath 
 * @param {string} backupFilePath 
 * @param {string} savePath 
 */
const writeTinyDataFile = (classes, trainFilePath, testFilePath, labelFilePath, backupFilePath, savePath) => {
    const content = `classes= ${classes}
train  = ${trainFilePath}
valid  = ${testFilePath}
labels = ${labelFilePath}
backup = ${backupFilePath}`;
    try{
        fs.writeFileSync(savePath, content);
        return null;
    }
    catch(err){
        throw err;
    }
}

/**
 * 
 * @param {string} dirPath 
 * @param {string} saveFilePath 
 */
const writeTrainingList = (dirPath, saveFilePath) => {
    const files = fs.readdirSync(dirPath);
    let content = "";
    files.forEach(file =>{
        content += path.resolve(dirPath , file + "\n");
    })
    try{
        fs.writeFileSync(saveFilePath, content);
        return null;
    }
    catch(err){
        throw err;
    }
}

module.exports = {
    randomInt: randomInt,
    writeTinyDataFile: writeTinyDataFile,
    writeTrainingList: writeTrainingList
}
