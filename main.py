import os

from algo.run_algo import run_algo
from utils.config import Config
from utils.logger import LoggerSingleton

# Get the pair list from cached_data folder
pair_list = [filename.split(".")[0] for filename in os.listdir("./cached_data/15m")]
pair_count = len(pair_list)

for pair_counter, pair_name in enumerate(pair_list, start=1):
    # Set the pair_name in the configuration singleton and the logger
    Config.set_pair_name(pair_name)
    LoggerSingleton.update_pair_name("positions", pair_name)
    LoggerSingleton.update_pair_name("ho_zigzag", pair_name)

    print(f"Processing pair {pair_counter}/{pair_count}:", pair_name)

    try:
        run_algo(pair_name, "15m")

    except Exception as e:
        print(f"Error processing {pair_name}: {e}")
        continue

