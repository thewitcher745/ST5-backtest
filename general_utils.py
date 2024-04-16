import pandas as pd
import plotly.graph_objects as go


def load_local_data(hdf_path: str = './cached_data/XEMUSDT.hdf5') -> pd.DataFrame:
    # Use pandas to read the hdf5 file
    return pd.DataFrame(pd.read_hdf(hdf_path))


def plot_candlestick_with_peaks(df: pd.DataFrame, zigzag_df: pd.DataFrame=None, x_axis_type="time"):
    fig = go.Figure()

    if x_axis_type == "index":
        candlestick_x_data = df.index
        zigzag_x_data = df.pair_df_index
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

    if zigzag_df is not None:
        fig.add_trace(go.Scatter(x=zigzag_x_data,
                                 y=zigzag_df.pivot_value,
                                 mode='lines+markers',
                                 name='Zigzag version 2',
                                 line=dict(color='royalblue')))

    fig.update_layout(title='Candlestick Chart with Peaks',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    fig.show()
