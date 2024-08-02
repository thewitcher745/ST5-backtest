from typing import Union, List, Tuple
import pandas as pd
import plotly.graph_objects as go

# from algorithm_utils import Box


def load_local_data(hdf_path: str = './cached_data/XEMUSDT.hdf5') -> pd.DataFrame:
    pair_df = pd.DataFrame(pd.read_hdf(hdf_path))
    pair_df['candle_color'] = pair_df.apply(lambda row: 'green' if row.close > row.open else 'red', axis=1)
    # Use pandas to read the hdf5 file
    return pair_df


class PlottingTool:
    def __init__(self, x_axis_type='index'):
        self.fig = go.Figure()
        self.x_axis_type = x_axis_type
        self.number_drawn_ranges = 0
        self.range_color_list = ['LightSalmon',
                                 'LightGreen',
                                 'LightBlue',
                                 'LightCoral',
                                 'LightSkyBlue',
                                 'LightPink',
                                 'LightYellow',
                                 'LightGray',
                                 'LightCyan',
                                 'LightGoldenrod',
                                 'LightSeaGreen']

    def draw_candlesticks(self, df) -> None:
        # Set the candlestick data to be plotted from the time values instead of pair_df_indices if the x_axis_type is time
        candlestick_x_data = df.index
        if self.x_axis_type == 'time':
            candlestick_x_data = df.time

        self.fig.add_trace(go.Candlestick(x=candlestick_x_data,
                                          open=df['open'],
                                          high=df['high'],
                                          low=df['low'],
                                          close=df['close'],
                                          name='Candlestick')
                           )

    def draw_zigzag(self, zigzag_df, title='Zigzag', color='royalblue') -> None:
        # Set the zigzag data to be plotted from the time values instead of pair_df_indices if the x_axis_type is time

        zigzag_x_data = zigzag_df.pair_df_index
        if self.x_axis_type == 'time':
            zigzag_x_data = zigzag_df.time

        # Plot the zigzag with the entered or default parameters
        self.fig.add_trace(go.Scatter(x=zigzag_x_data,
                                      y=zigzag_df.pivot_value,
                                      mode='lines+markers',
                                      name=title,
                                      line=dict(color=color)))

    def draw_highlight(self, highlight_range) -> None:
        # If a single highlight range has been given, draw it
        if type(highlight_range) == tuple:
            # Add one to the number of ranges drawn, this variable is used to keep colors separated
            # If there are no more colors, loop back to the first one
            self.number_drawn_ranges += 1
            next_color = self.range_color_list[self.number_drawn_ranges % len(self.range_color_list) - 1]

            self.fig.add_shape(
                type='rect',
                xref='x',
                yref='paper',
                x0=highlight_range[0],
                y0=0,
                x1=highlight_range[1],
                y1=1,
                fillcolor=next_color,
                opacity=0.5,
                layer='below',
                line_width=0,
            )
        elif type(highlight_range) == list:
            for i, range_tuple in enumerate(highlight_range):
                self.number_drawn_ranges += 1
                next_color = next_color = self.range_color_list[self.number_drawn_ranges % len(self.range_color_list) - 1]

                self.fig.add_shape(
                    type='rect',
                    xref='x',
                    yref='paper',
                    x0=range_tuple.start_index,
                    y0=0,
                    x1=range_tuple.end_index,
                    y1=1,
                    fillcolor=next_color,
                    opacity=0.5,
                    layer='below',
                    line_width=0,
                )

    def draw_points_with_label(self, x_data: list, y_data: list, label: str, color=None, draw_line=False):
        # Plot the zigzag with the entered or default parameters
        if not color:
            if label == "PBOS" or label == "BOS":
                color = "red"
            elif label == "LPL":
                color = "yellow",
            elif label == "LPLB":
                color = "orange"
            else:
                color = "purple"


        # positions = ["top center" if point_type == "peak" else "bottom center" for point_type in points_df.pivot_type.tolist()]

        self.fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+text' if draw_line else "markers+text",  # Added text mode
            name=label,
            marker=dict(
                color=color,
                size=20,
                symbol='circle-open'  # This creates hollow circles
            ),
            text=label,  # This sets the text to be displayed
            textposition="bottom center",  # This positions the text above the markers
            textfont=dict(size=10,
                          color=color)  # You can adjust the font size as needed
        ))

    def draw_box(self, box, pair_df_end_index, color=None):
        x1 = pair_df_end_index if len(box.price_reentry_indices) == 0 else box.price_reentry_indices[0]
        if color is None:
            color = "green" if box.type == "long" else "red"
        self.fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=box.start_index,
            y0=box.bottom,
            x1=x1,
            y1=box.top,
            fillcolor=color,
            opacity=0.5,
            layer="above",
            line_width=0,
        )

    def show(self, title: str = 'Price Chart',
             xaxis_title: str = 'Date',
             yaxis_title: str = 'Price'):

        self.fig.update_layout(title=title,
                               xaxis_title=xaxis_title,
                               yaxis_title=yaxis_title,
                               xaxis_rangeslider_visible=False)

        self.fig.show()

def convert_timestamp_to_readable(timestamp: pd.Timestamp):
    utc = timestamp.to_pydatetime()

    def two_char_long(num):
        if num >= 10:
            return str(num)
        else:
            return "0" + str(num)

    readable_format = f"{utc.year}.{utc.month}.{utc.day}/{two_char_long(utc.hour)}:{two_char_long(utc.minute)}:{two_char_long(utc.second)}"

    return readable_format