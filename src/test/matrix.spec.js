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

    it('should scale matrix values', () => {
        // truth
        let rows = helper.randomInt();
        let cols = helper.randomInt();
        let mat = yoloNodeJs.make_matrix(rows,cols);
        yoloNodeJs.scale_matrix(mat,5);
        expect(mat.vals).to.not.eq(undefined);
    })

    it('should resize matrix values', () => {
        let rows = helper.randomInt();
        let cols = helper.randomInt();
        let mat = yoloNodeJs.make_matrix(rows,cols);
        let resizerows = helper.randomInt();
        mat = yoloNodeJs.resize_matrix(mat, resizerows);
        expect(mat.vals).to.not.eq(undefined);
        expect(mat.vals).to.be.an("array");
        expect(mat.vals.length).to.be.eq(resizerows);
    })

    it('should add two matrix values', () => {
        let rows = helper.randomInt();
        let cols = helper.randomInt();
        let from = yoloNodeJs.make_matrix(rows,cols);
        let to = yoloNodeJs.make_matrix(rows,cols);
        yoloNodeJs.matrix_add_matrix(from, to);
        expect(from.vals).to.not.eq(undefined);
        expect(from.vals).to.be.an("array");
        expect(from.vals.length).to.be.eq(rows);
    })

    it('should copy matrix values', () => {
        let rows = helper.randomInt();
        let cols = helper.randomInt();
        let source = yoloNodeJs.make_matrix(rows,cols);
        let copy = yoloNodeJs.copy_matrix(source);
        expect(copy).to.not.eq(undefined);
        expect(copy.rows).to.be.eq(source.rows);
        expect(copy.cols).to.be.eq(source.cols);
        expect(copy.vals.length).to.be.eq(source.vals.length);
    })

    it('should hold out matrix values', () => {
        let rows = helper.randomInt();
        let cols = helper.randomInt();
        let source = yoloNodeJs.make_matrix(rows,cols);
        let hold = yoloNodeJs.hold_out_matrix(source, rows);
        expect(hold).to.not.eq(undefined);
    })

    it('should pop matrix column values', () => {
        let rows = helper.randomInt();
        let cols = helper.randomInt();
        let source = yoloNodeJs.make_matrix(rows,cols);
        let col = yoloNodeJs.pop_column(source, cols -1);
        expect(col).to.not.eq(undefined);
    })


})