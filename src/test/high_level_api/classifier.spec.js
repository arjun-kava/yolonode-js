const { expect } = require('chai');
const path = require('path');
const fs = require('fs');
const yoloNodeJs = require('../../yolonode');
const helper = require('../helper');

describe('Classify', () => {
    it('should intialize classifier', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
    })

    it('should set and get data file path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const dataFilePath = "/images/cifar_data.txt";
      classifier.dataFilePath = dataFilePath;
      expect(classifier.dataFilePath).to.not.eql(undefined);
      expect(classifier.dataFilePath).to.eql(dataFilePath);
    })

    it('should set and get cfg file path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const cfgFilePath = "/images/cifar.cfg";
      classifier.cfgFilePath = cfgFilePath;
      expect(classifier.cfgFilePath).to.not.eql(undefined);
      expect(classifier.cfgFilePath).to.eql(cfgFilePath);
    })

    it('should set and get weight file path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const weightFilePath = "/images/cifar.weights";
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
      const filePath = "images/dog.jpg";
      classifier.filePath = filePath;
      expect(classifier.filePath).to.not.eql(undefined);
      expect(classifier.filePath).to.eql(filePath);
    })

    it('should set and get labels path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const labelsPath = "images/label.txt";
      classifier.labelsPath = labelsPath;
      expect(classifier.labelsPath).to.not.eql(undefined);
      expect(classifier.labelsPath).to.eql(labelsPath);
    })

    it('should set and get trainning path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const trainListPath = "images/train.list";
      classifier.trainListPath = trainListPath;
      expect(classifier.trainListPath).to.not.eql(undefined);
      expect(classifier.trainListPath).to.eql(trainListPath);
    })

    it('should set and get testing path', () => {
      const classifier = yoloNodeJs.classifier();
      expect(classifier).to.not.eql(undefined);
      const testListPath = "images/test.list";
      classifier.testListPath = testListPath;
      expect(classifier.testListPath).to.not.eql(undefined);
      expect(classifier.testListPath).to.eql(testListPath);
    })
})