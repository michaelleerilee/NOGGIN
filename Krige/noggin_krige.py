#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Capture krige control parameters.')
parser.add_argument('-s','--sampling_fraction'\
                    ,dest='samplingFraction'\
                    ,metavar='samplingFraction'\
                    ,type=float, help='Suggested fraction of data to sample.')

parser.add_argument('-d','--input_directory'\
                    ,dest='inputDirectory'\
                    ,type=string\
                    ,help='The directory to search for input files.')

parser.add_argument('-o','--output_file'\
                    ,destf='outputFile'\
                    ,type=string\
                    ,help='The output file for the results')

# parser.add_argument('-a','--adapt_fraction'\
#                     ,dest='adaptFraction'\
#                     ,metavar='adaptFraction'\
#                     ,type=bool, help='Vary the sampling fraction over iterations.')

args=parser.parse_args()

if __name__ == '__main__':
    print('---noggin_krige start---')
    print('s: ',args.samplingFraction)
    print('---noggin_krige done---')
