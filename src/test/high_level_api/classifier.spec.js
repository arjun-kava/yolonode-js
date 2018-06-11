const { expect } = require('chai');
const path = require('path');
const fs = require('fs');
const yoloNodeJs = require('../../yolonode');
const helper = require('../helper');

const rootDir = __dirname;
const dataDir = `${rootDir}/../..`;
const dataPaths = {
  dataFilePath: path.resolve(`${dataDir}/data/cifar_small.data`),
  cfgFilePath:  path.resolve(`${dataDir}/data/cifar_small.cfg`),
  weightFilePath:  path.resolve(`${dataDir}/data/cifar_resume.weights`),
  filePath:  path.resolve(`${dataDir}/data/dog.jpg`),
  labelsPath:  path.resolve(`${dataDir}/data/labels.txt`),
  trainListPath:  path.resolve(`${dataDir}/data/train.list`),
  testListPath:  path.resolve(`${dataDir}/data/test.list`),
  resultDirPath:  path.resolve(`${dataDir}/data/backup`)
}

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

    it('should train classifier', () => {
      const classifier = yoloNodeJs.classifier();
      classifier.dataFilePath = dataPaths.dataFilePath;
      classifier.cfgFilePath = dataPaths.cfgFilePath;
      classifier.resultDirPath = dataPaths.resultDirPath;
      classifier.train();
    })
})