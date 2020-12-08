#!/usr/bin/python3
'''
This module is for universal arg parse
'''

import argparse


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--modelType', default='homegrown', help='choose from homegrown/baseline/resnet')
    parser.add_argument('-sp', '--splitType', default='random', help='choose from random/order')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('--D2', action='store_true', help='')
    parser.add_argument('-n', '--normalize', action='store_true', help='')
    parser.add_argument('-ds', '--dataSource', help='choose from neu/simu')
    opts = parser.parse_args()
    return opts
