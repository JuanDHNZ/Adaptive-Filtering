# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:01:38 2021

@author: Juan David
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--kaf', help='Filter to train')
parser.add_argument('--dataset', help='Dataset to use')
parser.add_argument('-N', help='Dataset length (if available)',default=1000,type=int)
parser.add_argument('-P', help='Grid size',default=4,type=int)

args = parser.parse_args()
kaf = args.kaf
db = args.dataset
n_samples = args.N
n_params = args.P

# kaf = "QKLMS_AMK"
# db = "lorenz"
# n_samples = 500
# n_params = 3

def main():
    from test_on_KAF import kafSearch
    df = kafSearch(kaf, db, n_samples, n_params)
    df.to_csv('GridSearchResults/' + kaf + '_' + db + '_' + str(n_samples) + '.csv')
    
if __name__ == "__main__":
    main()
