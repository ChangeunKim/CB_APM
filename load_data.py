import argparse
from utils.data_utils import get_data

import numpy as np
import pandas as pd

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--horizon", dest="horizon_return", action="store", type=int,
                        help="Prediction horizon in months (e.g., 1, 3, 6, 12)")
    args = parser.parse_args()

    # Generate dataset
    input, target = get_data(horizon_r=args.horizon_return)

    # Save data file as CSV files
    input.to_csv('data/input_' + str(args.horizon_return) + 'month.csv', index=False)
    target.to_csv('data/target_' + str(args.horizon_return) + 'month.csv', index=False)

if __name__=="__main__":
    main()