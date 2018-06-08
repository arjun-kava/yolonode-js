const { expect } = require('chai');
const path = require('path');
const fs = require('fs');
const yoloNodeJs = require('../yolonode');
const helper = require('./helper');

describe('Matrix', () => {
    it('should call make_matrix() function', () => {
        let rows = helper.randomInt();
        let cols = helper.randomInt();
        let matrix = yoloNodeJs.make_matrix(rows,cols);
        expect(matrix).to.not.eql(undefined);
        expect(matrix).to.be.an('object');
        expect(matrix).to.have.property('rows');
        expect(matrix).to.have.property('cols');
        expect(matrix).to.have.property('vals');
        expect(matrix.rows).to.eql(rows);
        expect(matrix.cols).to.eql(cols);
        expect(matrix.vals).to.be.an("array");
        //expect(matrix.vals).to.eql([[0, 0],[ 0, 0]]);
    })

    it('should find top k accuracy of matrix', () => {
        // truth
        let rows = helper.randomInt();
        let cols = helper.randomInt();
        let truth = yoloNodeJs.make_matrix(rows,cols);

        // guess
        let guess = yoloNodeJs.make_matrix(rows,cols);

        let k =1;
        const topk = yoloNodeJs.matrix_topk_accuracy(truth, guess, k);
        expect(topk).to.not.eq(undefined);
    })
})