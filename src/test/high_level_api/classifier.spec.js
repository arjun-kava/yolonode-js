const { expect } = require('chai');
const path = require('path');
const fs = require('fs');
const yoloNodeJs = require('../../yolonode');
const helper = require('../helper');

const rootDir = __dirname;
const dataDir = `${rootDir}/../..`;
const dataPaths = {
  trainDir: path.resolve(`${dataDir}/data/tiny/train`),
  testDir: path.resolve(`${dataDir}/data/tiny/test`),

  // train
  dataFilePath: path.resolve(`${dataDir}/data/tiny/tiny.data`),
  cfgFilePath:  path.resolve(`${dataDir}/data/tiny/tiny.cfg`),
  weightFilePath:  path.resolve(`${dataDir}/data/backup/tiny.weights`),
  filePath:  path.resolve(`${dataDir}/data/person.jpg`),
  labelsPath:  path.resolve(`${dataDir}/data/tiny/labels.txt`),
  trainListPath:  path.resolve(`${dataDir}/data/tiny/train.list`),
  testListPath:  path.resolve(`${dataDir}/data/tiny/test.list`),
  resultDirPath:  path.resolve(`${dataDir}/data/backup`),
  top: 5,
}

before( () =>{
  // writing trainning list
  helper.writeTrainingList(
    dataPaths.trainDir,
    dataPaths.trainListPath
  );
  helper.writeTrainingList(
    dataPaths.testDir,
    dataPaths.testListPath
  );

  // write data file
  helper.writeTinyDataFile(
    2, 
    dataPaths.trainListPath,
    dataPaths.testListPath,
    dataPaths.labelsPath,
    dataPaths.resultDirPath,
    dataPaths.dataFilePath);
})

describe('Classify', () => {
    it('should intialize classifier', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
    })

    it('should set and get data file path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const dataFilePath = dataPaths.dataFilePath;
      classifier.dataFilePath = dataFilePath;
      expect(classifier.dataFilePath).to.not.eql(undefined);
      expect(classifier.dataFilePath).to.eql(dataFilePath);
    })

    it('should set and get cfg file path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const cfgFilePath = dataPaths.cfgFilePath;
      classifier.cfgFilePath = cfgFilePath;
      expect(classifier.cfgFilePath).to.not.eql(undefined);
      expect(classifier.cfgFilePath).to.eql(cfgFilePath);
    })

    it('should set and get weight file path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const weightFilePath = dataPaths.weightFilePath;
      classifier.weightFilePath = weightFilePath;
      expect(classifier.weightFilePath).to.not.eql(undefined);
      expect(classifier.weightFilePath).to.eql(weightFilePath);
    })

    it('should set and get gpu', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const gpu = 1;
      classifier.gpu = gpu;
      expect(classifier.gpu).to.not.eql(undefined);
      expect(classifier.gpu).to.eql(gpu);
    })

    it('should set and get gpu list', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const gpusList = "1,2,3,4";
      classifier.gpusList = gpusList;
      expect(classifier.gpusList).to.not.eql(undefined);
      expect(classifier.gpusList).to.eql(gpusList);
    })

    it('should set and get file path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const filePath = dataPaths.filePath;
      classifier.filePath = filePath;
      expect(classifier.filePath).to.not.eql(undefined);
      expect(classifier.filePath).to.eql(filePath);
    })

    it('should set and get labels path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const labelsPath = dataPaths.labelsPath;
      classifier.labelsPath = labelsPath;
      expect(classifier.labelsPath).to.not.eql(undefined);
      expect(classifier.labelsPath).to.eql(labelsPath);
    })

    it('should set and get trainning path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const trainListPath = dataPaths.trainListPath;
      classifier.trainListPath = trainListPath;
      expect(classifier.trainListPath).to.not.eql(undefined);
      expect(classifier.trainListPath).to.eql(trainListPath);
    })

    it('should set and get testing path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const testListPath = dataPaths.testListPath;
      classifier.testListPath = testListPath;
      expect(classifier.testListPath).to.not.eql(undefined);
      expect(classifier.testListPath).to.eql(testListPath);
    })

    it('should set and get backup path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const resultDirPath = dataPaths.resultDirPath;
      classifier.resultDirPath = resultDirPath;
      expect(classifier.resultDirPath).to.not.eql(undefined);
      expect(classifier.resultDirPath).to.eql(resultDirPath);
    })

    it('should set and get top', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const top = 5;
      classifier.top = top;
      expect(classifier.top).to.not.eql(undefined);
      expect(classifier.top).to.eql(top);
    })

    it('should set and get thresh', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const thresh = .5;
      classifier.thresh = thresh;
      expect(classifier.thresh).to.not.eql(undefined);
      expect(classifier.thresh).to.eql(thresh);
    })

    it('should set and get hier thresh', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const hierThresh = .5;
      classifier.hierThresh = hierThresh;
      expect(classifier.hierThresh).to.not.eql(undefined);
      expect(classifier.hierThresh).to.eql(hierThresh);
    })

    it('should train classifier', () => {
      const classifier = yoloNodeJs.classifier();
      classifier.weightFilePath = dataPaths.weightFilePath;
      classifier.dataFilePath = dataPaths.dataFilePath;
      classifier.cfgFilePath = dataPaths.cfgFilePath;
      classifier.resultDirPath = dataPaths.resultDirPath;
      classifier.train();
    })

    /*it('should classify image tiny', () => {
      const classifier = yoloNodeJs.classifier();
      
      classifier.cfgFilePath = dataPaths.cfgFilePath;
      classifier.weightFilePath = dataPaths.testWeightPath;
      classifier.labelsPath = dataPaths.labelsPath;
      
      classifier.top = dataPaths.top;
      classifier.loadWeights();

      const testFiles = fs.readdirSync(dataPaths.testDir);
      for(let i = 0; i< 10; i++){
        classifier.filePath = path.resolve(dataPaths.testDir, testFiles[i]);
        classifier.classify();
      }
    })*/
})