const { expect } = require('chai');
const path = require('path');
const fs = require('fs');
const yoloNodeJs = require('../../yolonode');

describe('Utils', () => {
    it('should call what_time_is_it_now() function', () => {
        expect(yoloNodeJs.what_time_is_it_now()).to.not.eql(undefined);
    })
})