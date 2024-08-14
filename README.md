# ST5 Backtest
#### A strategy based on zigzags, normal ones and higher order ones

## Setting Up

1. Create a folder named `cached_data` in the root directory of the project.

2. Place your candlestick data files in the `cached_data` folder. The data files should be in HDF5 format. They should contain time and OHLC data for the asset you want to backtest.

## Running the Project

1. Ensure that you have Python installed on your system. This project requires Python 3.

2. Run the project by launching the ```main.ipynb``` Jupyter notebook.

## Changelog

### ver b0.2
- Algorithm now uses entirely new way of finding a higher order zigzag
- PBOS's get updated once a candle breaks it with a shadow. The search stops once a candle breaks it by CLOSING, aka below/above it.
- The region between the initial PBOS and the closing candle needs a low point, which is programmed to be the highest peak/lowest valley in the region. THe higher order zigzag is the constructed by connecting the first PBOS, the newly found inverse pivot in the middle region and the PBOS formed by the last LPL before the closing candle.
- The region search can also be broken by a candle which breaks the most recent low before a peak PBOS/high before a valley PBOS, aka the LPL which formed the initial PBOS. This causes a reversal in the direction.

### ver b0.3
- Continued overhaul of the algorithm, major updates and bits and pieces to make operation easier
- Now pattern can be extended without Change of Character (CHOCH), which will be implemented later, this means that consecutive patterns can form without any user input.
- Plotting utilities, extra arguments for most functions to account for testing applications

### ver b0.4
- LPL detection now requires higher highs AND higher lows instead of only higher lows (For asxcending patterns)
- Broken LPL's also updated to match
- Updated breaking sentiment detection, implemented CHOCH detection
- Updated main loop to now run indefinitely, until stopped by an error or when no closing candles are found
- Temporarily disabled pattern_count functionality