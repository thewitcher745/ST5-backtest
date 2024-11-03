from typing import Union, List, Tuple
import pandas as pd
import plotly.graph_objects as go
import constants


def load_local_data(pair_name: str = "BTCUSDT", timeframe: str = "15m") -> pd.DataFrame:
    hdf_path: str = f"./cached_data/{timeframe}/{pair_name}.hdf5"

    pair_df = pd.DataFrame(pd.read_hdf(hdf_path))
    pair_df['candle_color'] = pair_df.apply(lambda row: 'green' if row.close > row.open else 'red', axis=1)

    return pair_df


def load_higher_tf_data(pair_name: str = "BTCUSDT", timeframe: str = "4h") -> pd.DataFrame:
    hdf_path: str = f"./cached_data/{timeframe}/{pair_name}.hdf5"

    pair_df = pd.DataFrame(pd.read_hdf(hdf_path))
    pair_df['candle_color'] = pair_df.apply(lambda row: 'green' if row.close > row.open else 'red', axis=1)

    return pair_df


class PlottingTool:
    def __init__(self, x_axis_type='index', width=800, height=600):
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

        # Set the size of the plotting frame
        self.fig.update_layout(
            width=width,
            height=height
        )

    def save_plot(self, scale=1):
        self.fig.write_image("./plot.png", format="png", scale=scale)

    def zoom_on_candle(self, candle_index, y_zoom_range: list = None, x_zoom_range=50) -> None:
        """
        This functions zooms on a specific candle in the plotting window.

        Parameters:
        candle_index (int): The index of the candle to zoom on.
        y_zoom_range (list): The range of the y-axis to zoom in on. If None, the y-axis will not be zoomed in, absolute
        x_zoom_range (int): The range of the x-axis to zoom in on from both directions, relative

        """
        self.fig.update_xaxes(range=[candle_index - x_zoom_range, candle_index + x_zoom_range])

        if y_zoom_range:
            self.fig.update_yaxes(range=y_zoom_range)

    def draw_candlesticks(self, df, label="Candlestick") -> None:
        # Set the candlestick data to be plotted from the time values instead of pair_df_indices if the x_axis_type is time
        candlestick_x_data = df.index
        if self.x_axis_type == 'time':
            candlestick_x_data = df.time

        self.fig.add_trace(go.Candlestick(x=candlestick_x_data,
                                          open=df['open'],
                                          high=df['high'],
                                          low=df['low'],
                                          close=df['close'],
                                          name=label)
                           )

    def draw_zigzag(self, zigzag_df, title='Zigzag', color='royalblue') -> None:
        # Set the zigzag data to be plotted from the time values instead of pair_df_indices if the x_axis_type is time

        zigzag_x_data = zigzag_df.pdi
        if self.x_axis_type == 'time':
            zigzag_x_data = zigzag_df.time

        # Plot the zigzag with the entered or default parameters
        self.fig.add_trace(go.Scatter(x=zigzag_x_data,
                                      y=zigzag_df.pivot_value,
                                      mode='lines',
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

    def draw_points_with_label(self, x_data: list, y_data: list, label: str = "", color="black", draw_line=False):
        # Plot the zigzag with the entered or default parameters
        # if not color:
        #     if label == "PBOS" or label == "BOS":
        #         color = "red"
        #     elif label == "LPL":
        #         color = "yellow",
        #     elif label == "LPLB":
        #         color = "orange"
        #     else:
        #         color = "purple"

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

    def draw_box(self, box, pair_df_end_index=None, force_end_pdi=False, color=None):
        x1 = pair_df_end_index if len(box.price_reentry_indices) == 0 else box.price_reentry_indices[0]

        if force_end_pdi:
            x1 = pair_df_end_index
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

    def draw_fvg_box(self, fvg):
        self.fig.add_shape(type="rect",
                           x0=fvg.middle_candle - 1,
                           y0=fvg.fvg_lower,
                           x1=fvg.middle_candle + 1,
                           y1=fvg.fvg_upper,
                           fillcolor="blue",
                           opacity=0.5,
                           name="FVG")

    def draw_segment_bbox(self, segment, color="yellow"):
        """
        This method draws a hollow blue bounding box for a segment that will be processed in the backtest.
        Args:
            segment: The segment to be drawn
        """
        self.fig.add_shape(type="rect",
                           x0=segment.start_pdi,
                           y0=segment.bottom_price,
                           x1=segment.end_pdi,
                           y1=segment.top_price,
                           line=dict(
                               width=2,  # Border width
                               color=color  # Border color
                           ),
                           name="Segment")

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


def find_higher_timeframe(lower_timeframe):
    for i, key in enumerate(constants.timeframe_minutes.keys()):
        if key == lower_timeframe:
            return list(constants.timeframe_minutes.keys())[i + constants.higher_timeframe_interval]
