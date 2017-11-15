#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")

args = parser.parse_args()
class compare_robot(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.fileDir = os.path.dirname(os.path.abspath("../openface/demos/compare.py"))
        self.modelDir = os.path.join(self.fileDir, '..', 'models')
        self.dlibModelDir = os.path.join(self.modelDir, 'dlib')
        self.openfaceModelDir = os.path.join(self.modelDir, 'openface')

        self.dlibFacePredictor = os.path.join(self.dlibModelDir, "shape_predictor_68_face_landmarks.dat")
        self.networkModel = os.path.join(self.openfaceModelDir, 'nn4.small2.v1.t7')
        self.imgDim = 96

        self.align = openface.AlignDlib(self.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.networkModel, self.imgDim)
        

    def getRep(self, imgPath):
        if self.verbose:
            print("Processing {}.".format(imgPath))
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.verbose:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        bb = self.align.getLargestFaceBoundingBox(rgbImg)
        if bb is None:
            raise Exception("Unable to find a face: {}".format(imgPath))
        if self.verbose:
            print("  + Face detection took {} seconds.".format(time.time() - start))

        start = time.time()
        alignedFace = self.align.align(self.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        if self.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        rep = self.net.forward(alignedFace)
        if self.verbose:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
            print("Representation:")
            print(rep)
            print("-----\n")
        return rep


def main():
    start = time.time()
    cbot = compare_robot()
    for (img1, img2) in itertools.combinations(args.imgs, 2):
        d = cbot.getRep(img1) - cbot.getRep(img2)
        print("Comparing {} with {}.".format(img1, img2))
        print(
            "  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))

if __name__ == "__main__":
    main()
