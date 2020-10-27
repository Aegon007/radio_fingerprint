#!/usr/bin/python3

import os
import sys
import argparse
import pdb


ROOT_DIR = os.getenv('ROOT_DIR')


def main(opts):
    pass


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument()
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseArgs(sys.argv)
    main(opts)
