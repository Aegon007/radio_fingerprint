#!/usr/bin/python3
'''
This module is for universal arg parse
'''

import argparse


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--modelType', help='choose from homegrown/baseline/resnet')
    parser.add_argument('-sp', '--splitType', help='choose from random/order')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    opts = parser.parse_args()
    return opts
