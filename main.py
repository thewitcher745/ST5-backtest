import os

from algo.run_algo import run_algo

# Get the pair list from cached_data folder
pair_list = [filename.split(".")[0] for filename in os.listdir("./cached_data/15m")]

for pair_name in pair_list:
    run_algo(pair_name, "15m")

