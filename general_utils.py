from typing import Union, List, Tuple
import pandas as pd
import plotly.graph_objects as go


def load_local_data(hdf_path: str = './cached_data/XEMUSDT.hdf5') -> pd.DataFrame:
    # Use pandas to read the hdf5 file
    return pd.DataFrame(pd.read_hdf(hdf_path))


def plot_candlesticks_zigzag_range(df: pd.DataFrame,
                                   zigzag_df: pd.DataFrame = None,
                                   x_axis_type: str = 'time',
                                   # highlight_range: Union[List[PatternTupleType], tuple] = None,
                                   directional_coloring: bool = True) -> None:
    """
        Function to plot candlestick chart with optional zigzag lines and highlighted ranges.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the candlestick data.
        zigzag_df (pd.DataFrame, optional): The DataFrame containing the zigzag data. Default is None.
        x_axis_type (str, optional): The type of x-axis. Can be 'time' or 'index'. Default is 'time'.
        highlight_range (Union[List[Tuple]], optional): The range(s) to be highlighted on the chart. Default is None.
        directional_coloring (bool, optional): If True, the highlighted range will be colored based on the direction. Default is True.

        Returns:
        None: The function directly plots the chart.
    """

    fig = go.Figure()

    # Make the x-axis based on 'time' or 'index'
    if x_axis_type == 'index':
        candlestick_x_data = df.index
        zigzag_x_data = zigzag_df.pair_df_index
    else:
        candlestick_x_data = df.time
        zigzag_x_data = zigzag_df.time

    # Add the candlestick chart
    fig.add_trace(go.Candlestick(x=candlestick_x_data,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Candlestick'))

    # Check if the zigzag data has been given
    if zigzag_df is not None:
        fig.add_trace(go.Scatter(x=zigzag_x_data,
                                 y=zigzag_df.pivot_value,
                                 mode='lines+markers',
                                 name='Zigzag version 2',
                                 line=dict(color='royalblue')))

    # Add a highlighted range if specified
    # if highlight_range is not None:
    #     # If a single highlight range has been given, draw it
    #     if type(highlight_range) == tuple:
    #         fig.add_shape(
    #             type='rect',
    #             xref='x',
    #             yref='paper',
    #             x0=highlight_range[0],
    #             y0=0,
    #             x1=highlight_range[1],
    #             y1=1,
    #             fillcolor='LightSalmon',
    #             opacity=0.5,
    #             layer='below',
    #             line_width=0,
    #         )
    #     elif type(highlight_range) == list:
    #         for i, range_tuple in enumerate(highlight_range):
    #             if directional_coloring:
    #                 # If directional coloring is enabled, color the range based on the direction (Bearish or bullish)
    #                 # If enabled, the pattern_list has to have a length of 3, otherwise, throw an error
    #                 if len(highlight_range[i]) != 3:
    #                     raise ValueError('The highlight_range list has to be a list of tuples with 3 elements each')
    #                 fill_color = 'LightGreen' if highlight_range[i].type == 'bullish' else 'LightSalmon'
    #             else:
    #                 # A list of colors to loop through and draw patterns from
    #                 # If the list is exhausted, it will loop back to the beginning
    #                 color_list = ['LightSalmon', 'LightGreen', 'LightBlue', 'LightCoral', 'LightSkyBlue', 'LightPink']
    #                 fill_color = color_list[i % len(color_list)]
    #
    #             fig.add_shape(
    #                 type='rect',
    #                 xref='x',
    #                 yref='paper',
    #                 x0=range_tuple.start_index,
    #                 y0=0,
    #                 x1=range_tuple.end_index,
    #                 y1=1,
    #                 fillcolor=fill_color,
    #                 opacity=0.5,
    #                 layer='below',
    #                 line_width=0,
    #             )

    fig.update_layout(title='Candlestick Chart with Peaks',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    fig.show()
