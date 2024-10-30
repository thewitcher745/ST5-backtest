# ST5 Backtest

#### A strategy based on zigzags, normal ones and higher order ones

## Setting Up

1. Create a folder named `cached_data` in the root directory of the project.

2. Place your candlestick data files in the `cached_data` folder. The data files should be in HDF5 format. They should contain time and OHLC data for
   the asset you want to backtest.

## Running the Project

1. Ensure that you have Python installed on your system. This project requires Python 3.

2. Run the project by launching the ```main.ipynb``` Jupyter notebook.

## Changelog

### ver b0.2

- Algorithm now uses entirely new way of finding a higher order zigzag
- PBOS's get updated once a candle breaks it with a shadow. The search stops once a candle breaks it by CLOSING, aka below/above it.
- The region between the initial PBOS and the closing candle needs a low point, which is programmed to be the highest peak/lowest valley in the
  region. THe higher order zigzag is the constructed by connecting the first PBOS, the newly found inverse pivot in the middle region and the PBOS
  formed by the last LPL before the closing candle.
- The region search can also be broken by a candle which breaks the most recent low before a peak PBOS/high before a valley PBOS, aka the LPL which
  formed the initial PBOS. This causes a reversal in the direction.

### ver b0.3

- Continued overhaul of the algorithm, major updates and bits and pieces to make operation easier
- Now pattern can be extended without Change of Character (CHOCH), which will be implemented later, this means that consecutive patterns can form
  without any user input.
- Plotting utilities, extra arguments for most functions to account for testing applications

### ver b0.4

- LPL detection now requires higher highs AND higher lows instead of only higher lows (For asxcending patterns)
- Broken LPL's also updated to match
- Updated breaking sentiment detection, implemented CHOCH detection
- Updated main loop to now run indefinitely, until stopped by an error or when no closing candles are found
- Temporarily disabled pattern_count functionality

### ver b0.41

- Broken LPL detection now omits consecutive broken LPL's, and only registers the last on in the chain
- Finding the last HO zigzag pivot in the mid region now uses a different method. Previously it extended the last LPL in the region and selected a new
  pivot right after. In the updated version, the new ending pivot is selected by finding the lowest point between the mid-region pivot and the first "
  breaking" LPL (of the correct type) after the closing candle

### ver b0.5

- Broken LPL detection now uses completely new algorithm. most recent highs and lows are recorded, an LPL is registered only when a highest high is
  broken. if a lowest low is broken, it is recorded as a broken LPL and a completely new LPL chain is initialized.
- Pattern continuation changed, now a BOS update causes an extremum (lowest low or highest high) and the pattern restarts from there. A CHOCH causes
  the pattern to restart with the last BOS being the starting point, effectively flipping the direction.
- Logging implemented, now can output to file and stdout with settable allowed verbosity levels in constants.py
- Broken LPL algorithm changed and optimized, now doesn't try to detect all broken LPL's, only starts at one point and detects the first broken LPL.
  This causes the search to only continue up to the point that we need, not further, majorly improving performance.
- detect_breaking_statement now also accounts for CHOCH_SHADOW case for extending and updating the CHOCH threshold from the original HO zigzag pivot.
  Also improved the performance and simplified the logic of the function.

### ver b0.6

- Automated starting point detection implemented. A constant in constants.py governs how many candles in a higher timeframe we backtrack and start
  drawing a higher order zigzag on. Then, the last higher order pivot on the higher timeframe BEFORE the original starting point of the lower
  timeframe is selected as the starting point.
- The type of the pivot (peak/valley) determines exactly which point to start from in the lower timeframe data. This means if the last pivot type is
  a "low" (valley), the lowest low in the n candles is selected (Parameter "n" being the number of lower order candles being aggregated in the higher
  order timeframe candle, which is determined using a simple mathematical formula/dict)
- The starting point is then used to draw the higher order zigzag, and the algorithm continues as normal. This is a major update and a big step
  towards full automation of the algorithm.

### ver b0.7

- Order blocks implemented. The class used to be named "Box" and implemented in the datatypes.py module. It's now moved to algorithm_utils but might
  later be moved as a cleaner solution.
- Order blocks now store basic information about the formation of the OB, such as the starting index, top and bottom, price exit and reentry, etc.
- Rudimentary features for performing condition checks were also implemented, mainly a function to form the window to perform the checks on.
- Plotting utilities updated to include zooming functionalities, on x and y-axis, on a specific candle specified by an index.
- Lower order zigzags are now stripped of their markers to provide a tidier chart.

#### ver b0.71

- FVG and block stop break checks implemented. These checks are made using two methods in the OrderBlock method. FVG and block stop break check range
  is from the starting base candle to the next pivot (next high for ascending pattern and next low for descending pattern)

### ver b0.8

- Segments and their respective OB's implemented. Segments are formed in each region where a PBOS_CLOSE event takes place. The segment starts at the
  pivot before the PBOS and ends on the candle before the candle that closes above/below the PBOS, making it a BOS.
- The significance of segments is that in the candles within them, the order block aren't updated or invalidated or modified in any way. So in a
  certain section of a segment, we can safely check for entries to the order blocks within that segment without having to worry about OB updates.
- A segment's order blocks form on its first leg only, the first leg being the pivot before the PBOS to the PBOS itself. This means that only the
  lower order lows (Ascending patterns) or lower order highs (Descending patterns) are used to form the OB's. This is to prevent the OB's from being
  formed on the opposite-direction trend, which goes against the concept of the strategy.
- A segment's order blocks can only register entries after a certain point in the candles within the segment. This certain point is exactly the lower
  order pivot that breaks the LPL, forming the initial PBOS. This is to prevent look-ahead bias, as the order blocks wouldn't have formed in the first
  place if the LPL hadn't been broken. Therefore only order blocks that have their entries after this value and before the end of the segments are
  considered eligible for entry registration.
- Plotting tool updated to include segment plotting, which is bounding box with a distinguishable color.
- Order block entry registry implemented. This method uses a constant value for the used capital which is queries from the constants.py file
  currently. This can later be changed to include dynamic capital allocation and much more.

### ver b0.9

- Position exiting fully implemented. Positions can now exit when they either achieve FULL_TARGET status or when they hit the stop loss.
- Before a stoploss or a full target event, the highest target hit is calculated and registered. Each target hit time is registered in the Position
  object instance for later validation
- Rudimentary report generation using excel files implemented. The report contains data on the final status of positions, their types, net profits,
  entry and exit times, entry and exit prices, targets hit, and most other usefulvalidation sources.
- The exiting and report generation code is currently implemented directly in the main Jupyter notebook, but will later be moved to a separate module
  for better organization and readability.

#### ver b0.91

- Order block exit candle check redesigned. Now for a long position, the exit candle should open inside the OB and close outside of it, instead of
  just having a high value outside the OB.
- Checking for broken LPL now does NOT skip the new starting point.

#### ver b0.92

- check_fvg_condition now returns False if a proper exit candle (According to ver b0.91) isn't found.
- Added "Ranking within segment" property to order blocks. This property stores the ranking of the order block within its segment. This is a
  sequential number which tells us what position the order block holds within all the order blocks in that segment.
- Added (and temporarily removed) a "Has been replaced?" property to order blocks which could potentially store the order blocks that were replaced (
  due to the FVG condition not being satisfied) to the output report. This made it easier to track if the replacement process is actually worth it or
  not.

#### ver b0.10

- Now the algorithm will also try to form segments when a CHOCH_CLOSE event happens. Previously, the algorithm actually had look-ahead bias as it only
  formed positions on PBOS_CLOSE events, before the actual outcome was known. Now, the algorithm will form segments on CHOCH_CLOSE events as well.
- The segments now have a formation_method property which tells us how the segment was formed. This is useful for debugging and validation purposes,
  and also used in the plotting of the segments.

#### ver b0.11: The FIND_ORDER_BLOCKS update!

- algorithm_utils now uses a robust logging module (based on the logging library) to log position formation-related information. The logged output
  includes tabbed (horizontal) and vertical spacing for super easy readability. The HO zigzag logs will later be migrated to use this module as well.
- Finding the location where the LPL breaks now uses a better method of finding the breaking candle. Previously, the search was only performed on LO
  zigzag pivots, and the PDI of whichever one broke it first was returned as the breaking candle PDI. Now the algorithm actually looks for the candle
  that broke the LPL, in a small region ranging from the breaking pivot's previous pivot (Normally a higher order pivot) to the breaking pivot itself.
  This region will be searched for a candle that breaks the breaking_value (According to the Algo.detect_first_broken_lpl() method) instead of a pivot
  that does.
- Checking the reentry condition has been overhauled. Now it has a separate checking method (OrderBlock.has_entry_condition) and a separate check
  flag (OrderBlock.has_reentry_condition, True by default). The method uses simple pandas dataframe slices to look for reentries in a
  reentry_check_window (passed as an argument). If a candle has pierced the top (long OB) or bottom (short OB) of an order block in this reentry check
  window, the check will fail and the flag will be set to False. the reentry_check_window will be passed to the function through the
  Segment.find_order_blocks method (Explained further in the changelog)
- The Segment.find_order_blocks, which was a method which, within each segment, attempted to find valid order blocks, now uses the new reentry check
  method. Also, now if an OB doesn't have a valid exit candle, the search will continue and the next candle will be checked.
- OrderBlock.check_box_entries which registered the reentries (naively and unnecessarily, will later be revised and formatted for cleaner code) and
  more importantly the order block's exit candle, has been reformatted to only look for the exit candle up to a certain upper bound, passed as an
  argument (upper_search_bound_pdi). Currently, the upper bound is passed in the Segment.find_order_blocks() method, and is set to the end of the
  segment. This means the search for the exit candle will only continue up to the last candle of the segment and no further. This can probably be
  optimized, but it does make sense since the order block can only enter before the segment ends, and the exit candle should be found by then. If an
  exit candle is not found in this region, the price_exit_index property of the order block will remain as None, failing most of the checks for that
  OB.