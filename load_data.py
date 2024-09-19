import argparse
from utils.data_utils import get_data

import numpy as np
import pandas as pd

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--return", dest="horizon_return", action="store", type=int)
    parser.add_argument("-c", "--consensus", dest="horizon_consensus", action="store", type=int)
    args = parser.parse_args()

    # Generate dataset
    input, target = get_data(horizon_r=args.horizon_return, horizon_c=args.horizon_consensus)

    # Save data file as CSV files
    if args.horizon_consensus:
        input.to_csv('data/input_predict_' + str(args.horizon_consensus) + '_' + str(args.horizon_return) + 'month.csv', index=False)
        target.to_csv('data/target_predict_' + str(args.horizon_consensus) + '_' + str(args.horizon_return) + 'month.csv', index=False)
    else:
        input.to_csv('data/input_approx_' + str(args.horizon_return) + 'month.csv', index=False)
        target.to_csv('data/target_approx_' + str(args.horizon_return) + 'month.csv', index=False)


if __name__=="__main__":
    main()