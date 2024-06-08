"""
Bokeh application produces interactive plots of GLERL's historical Great Lakes
ice cover concentration data

Data source: https://www.glerl.noaa.gov/data/ice/

@author: Jacob Bruxer
"""

import pandas as pd
from datetime import datetime, timedelta
from bokeh.io import curdoc, show
from bokeh.models.sources import ColumnDataSource
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.models import FixedTicker, Range1d, HoverTool, Label, Div, Slider
from bokeh.layouts import layout, grid, gridplot, column, row
from math import pi

# Application is designed to be launched as a full Bokeh server application
# which will use an html template and css style sheet to display some custom
# formatting; However, if standalone application is desired, setting FULL_APP
# to False will produce an HTML outfile with some simplified formatting

FULL_APP = True  # set to False for standalone HTML app

PLOT_WIDTH = 350 if FULL_APP is True else 600
PLOT_HEIGHT = int(PLOT_WIDTH * 0.7)

SELECTED_COLOUR_1 = '#14e5ff'
SELECTED_COLOUR_2 = '#ff61a8'

# Season defined as starting on November 1, season_days as integers follow
SEASON_DAYS_1 = int(datetime(1900,11,1).strftime('%j'))

def get_current_yr_data(current_yr):
    """
    Fetches daily average ice concentration data for all Great Lakes

    Parameters
    ----------
    current_yr : int
        The current year to fetch the data.

    Returns
    -------
    current_df : pandas dataframe
        Current year's data, stored as day, value_lake1, value_lake2, ...

        Columns include:
        'sup', 'mic', 'hur', 'eri', 'ont', 'bas' (ie, full Great Lakes basin).


    """
    # fetch and read current year's data
    last_yr = current_yr - 1

    #url = f'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g{last_yr}_{current_yr}_ice.dat'
    url = f'https://apps.glerl.noaa.gov/coastwatch/webdata/statistic/ice/dat/g{last_yr}_{current_yr}_ice.dat'
        # convert names to same as historical data
    names=['year','day','sup','mic','hur','eri','ont','stc','bas']
    current_df = pd.read_csv(url, sep='\s+', skiprows=6, header=0, names=names)
        # for the current year's data, the dates are already in julian days
    julian = current_df['day'].iloc[0]

    # for plotting purposes, we set Nov 1 to season_day 1
    offset = julian-SEASON_DAYS_1
    days = list(range(offset, offset+len(current_df)))
    current_df['season_day'] = days
    current_df.set_index('season_day', inplace=True)
    
    return current_df


def get_historical_data(lake):
    """
    Fetches historical daily average ice cover concentration data for one lake

    Parameters
    ----------
    lake : string

        Name of lake data to fetch.  Options are:

        'sup', 'mic', 'hur', 'eri', 'ont', 'bas' (ie, full Great Lakes basin).

    Returns
    -------
    df : pandas dataframe

        Tabular time series of ice concentration data (% cover) with
        season_days (starting Nov 1) as index and years as columns.

        Example of formatting:

        season_day | 1973 | 1974 | ... | last year
        ----------------------------------
            1      |  0.0 |  0.0 | ... |    0.0
            2      |  1.2 |  0.0 | --- |    0.0
            3      |  2.5 |  0.8 | --- |    0.2
           ...
           220     |  0.0 |  0.0 | --- |    0.0

    """

    # fetch and read historical data through last year
    #url = f'''https://www.glerl.noaa.gov/data/ice/daily/{lake}.txt'''
    url = f'''https://www.glerl.noaa.gov/data/ice/glicd/daily/{lake}.txt'''

    # index is the date column to start
    hist_df = pd.read_csv(url, sep='\s+', index_col=[0])

    # this will drop the NaN for Feb-29 in non-leap years
    top = hist_df.loc[:"Feb-28"]  # split all data up to Feb-28
    bottom = hist_df.loc["Feb-29":]  # split remainder of data

    # will sort all Nan to bottom and retain order of real values
    # unlikely for this case, but an issue here in that if there are other NaN
    # values within the data they will also be pushed to bottom
    bottom = bottom.apply(lambda x: sorted(x, key=pd.isnull), axis=0)

    # re-combine top and bottom
    hist_df = pd.concat([top, bottom], sort=False)

    # convert string dates to julian days for plotting
    hist_df.reset_index(inplace=True)

    first_date = hist_df['index'].iloc[0]  # first date in table
    
    first_date = datetime.strptime(first_date, '%b-%d')
    julian = int(first_date.strftime('%j'))
    
    # drop date index
    hist_df.drop('index', axis=1, inplace=True)
    
    # for plotting purposes, we'll set Nov 1 to season_day 1
    offset = julian-SEASON_DAYS_1
    days = list(range(offset, offset+len(hist_df)))
    hist_df['season_day'] = days
    hist_df.set_index('season_day', inplace=True)

    return hist_df


def combine_data(lake, current_yr, current_df, hist_df):
    """
    Combine historical & current year's data into a single, complete dataframe

    Parameters
    ----------
    lake : string
        Name of lake data to fetch.  Options are:

        'sup', 'mic', 'hur', 'eri', 'ont', 'bas' (ie, full Great Lakes basin).

    current_yr : int
        Current year, only necessary to rename column.
    current_df : pandas dataframe
        Current year's data, stored as day, value series.
    hist_df : pandas dataframe
        Historical data, stored as table of day, value_yr1, value_yr2, ...

    Returns
    -------
    df : pandas dataframe
        Combined dataset.
        Stored as table of day, value_yr1, value_yr2... value_current_yr

    """

    # create full, clean empty dataframe
    df = pd.DataFrame(index=list(range(1,221))) # Nov 1 to Jun 10

    # concat with historical df
    df = pd.concat([df, hist_df], axis=1)

    # concat with current df
    df = pd.concat([df, current_df[lake]], axis=1)
    
    
    # *** ADDED 2024-06-08
    # *** If statement checks for duplicate columns for current year
    # *** Possible that historical data now also includes current year
    # *** However, unable to check since in June (post-winter season)
    # *** Check in winter season to confirm
    if str(current_yr) in df:
        
        df.drop(str(current_yr), axis=1, inplace=True)
    
    # rename column from current dataframe to current year
    df.rename({lake : str(current_yr)}, axis=1, inplace=True)
    #df.drop_duplicates(keep='last', inplace=True)

    # give index a name
    df.index.rename('season_day', inplace=True)
        
    return df


def create_baseplot(lake):
    """
    Create base Bokeh plot

    Parameters
    ----------
    lake : string
        Name of lake data to fetch.  Options are:

        'sup', 'mic', 'hur', 'eri', 'ont', 'bas' (ie, full Great Lakes basin).

    Returns
    -------
    fig : bokeh figure
        Empty Bokeh plot with minimal fomatting

    """

    plot_width = PLOT_WIDTH
    plot_height = PLOT_HEIGHT
    tools=[]

    fig = figure(title=lake,
                 plot_height=plot_height,
                 plot_width=plot_width,
                 tools=tools)

    # y-axis formatting
    fig.y_range=Range1d(0, 100, bounds='auto')
    fig.yaxis.axis_label = 'Ice cover (%)' # unicode deg Celsius

    # x-axis formatting
    fig.xaxis.major_label_orientation = pi/2
    fig.x_range=Range1d(0, 220, bounds='auto')

    # locate x-ticks at start of each month
    ticks = []
    day_of_year=0

    for days in [1,30,31,31,28,31,30,31,30]:
        day_of_year = day_of_year + days
        ticks.append(day_of_year)

    fig.xaxis.ticker = FixedTicker(ticks=ticks)

    # label ticks with month name
    labels = ['Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
    tick_labels = dict(zip(ticks, labels))
    fig.xaxis.major_label_overrides = tick_labels

    # Balances grid since it ends after June 1
    fig.x_range.start = -5

    fig.grid.grid_line_dash = [5,2]

    fig.toolbar.logo = None

    return fig


def plot_all(fig, df, start_year, end_year):
    """
    Add all years as lines to the Bokeh figure

    Parameters
    ----------
    fig : Bokeh figure
        Empty bokeh figure.
    df : pandas dataframe
        Tabular dataframe of ice data.
    start_year : int
        Starting year to plot.
    end_year : int
        Ending year to plot.

    Returns
    -------
    fig : Bokeh figure
        Same bokeh figure but now including all years plotted as lines.
    lines : list of bokeh line glyphs
        Plotted lines are assigned as variables and stored in list for
        use in legend later.

    """
    lines = list() # empty list to hold each plotted line

    # ensures ColumnDataSource will work
    df.columns = df.columns.astype(str)
    source = ColumnDataSource(df)

    # generate grey palette the size of dataset
    # but use offset at each end to avoid using some of the darkest and
    # lightest values
    offset = 10
    palette = palettes.grey(end_year - start_year + 1 + offset*4)

    unselected_kwargs = dict(line_width = 1.5,
                             line_alpha = 0.5)

    # plot all years
    for i, yr in enumerate(range(start_year, end_year+1)):

        lines.append(fig.line(x='season_day',
                              y=str(yr),
                              source=source,
                              color=palette[i+offset],
                              name=str(yr),
                              **unselected_kwargs))


    return fig, lines


def get_tooltips(year):
    """
    Generate a simple tool tip

    Parameters
    ----------
    year : int
        Year for tooltip.

    Returns
    -------
    tooltips : bokeh tooltip
        Formatted tooltip for use in bokeh figure.

    """

    try:
        year = str(year)
    except:
        pass

    tooltips = [('Date', '-'.join((year,'@date'))),
                ('Value', '@{}{}%'.format(year,'{0.0}'))]

    return tooltips


def plot_selected(fig, df, year, selected_colour):
    """
    Add a highlighted line for a selected year to the Bokeh figure

    Parameters
    ----------
    fig : Bokeh figure
        Empty bokeh figure.
    df : pandas dataframe
        Tabular dataframe of ice data.
    year : int
        Selected year to add highlighted line to plot.
    selected_colour : string
        Colour to use for plotting selected line.

    Returns
    -------
    fig : Bokeh figure
        Same bokeh figure but now including selected year plotted as
        highlighted line.
    selected : bokeh line glyph
        Plotted line is assigned as variable for use in legend later.
    hover : bokeh hover tool
        Hover tool with tooltip for selected line.
    label : bokeh label
        Label with year text.

    """
    selected_kwargs = dict(line_color = selected_colour,
                           line_width = 4)

    source = ColumnDataSource(df)

    y=str(year)
    selected = fig.line(x='season_day',
                          y=y,
                          source=source,
                          name=y,
                          **selected_kwargs)

    tooltips = get_tooltips(y)

    hover = fig.add_tools(HoverTool(tooltips=tooltips,
                          toggleable = False,
                          names=[y],
                          mode='mouse'
                          ))

    # find location of peak value for labelling
    labelx = df[str(year)].idxmax(axis=0) # index of max ice
    labely = df.loc[:, str(year)].max() # max ice
    
    # add label for selected year, locate at peak
    label = Label(x=labelx, y=labely, x_units='data',
                        text=f'{year}', render_mode='css',
                        text_color=selected_colour,
                        text_baseline='bottom',
                        text_font_style = 'bold')

    fig.add_layout(label)

    return fig, selected, label, hover


def build_layout():

    #
    current_year = datetime.now().year

    if datetime.now().month >= 10:
        current_year += 1

    # get current year's data for all of the Great Lakes
    current_df = get_current_yr_data(current_year)

    last_data_yr = current_df['year'].iloc[-1]
    last_data_day_of_yr = current_df['day'].iloc[-1]
    try:
        last_data_yr = int(last_data_yr)
        last_data_day_of_yr = int(last_data_day_of_yr)
    except:
        pass
    last_data_updated = datetime(last_data_yr,1,1)+timedelta(days=last_data_day_of_yr-1)

    # get historical data for each lake individually then combine with current year
    lakes = {'Great Lakes' : {'data' : combine_data('bas', current_year, current_df,
                                                     get_historical_data('bas'))},
             'Lake Superior': {'data' : combine_data('sup', current_year, current_df,
                                                     get_historical_data('sup'))},
             'Lake Michigan': {'data' : combine_data('mic', current_year, current_df,
                                                     get_historical_data('mic'))},
             'Lake Huron': {'data' : combine_data('hur', current_year, current_df,
                                                     get_historical_data('hur'))},
             'Lake Erie': {'data' : combine_data('eri', current_year, current_df,
                                                     get_historical_data('eri'))},
             'Lake Ontario': {'data' : combine_data('ont', current_year, current_df,
                                                     get_historical_data('ont'))}}


    for lake in lakes.keys():

        # create an empty plot
        fig = create_baseplot(lake)
        
        df = lakes[lake]['data']
        
        start_year = int(df.columns[0])
        end_year = int(df.columns[-1])
        
        # add dummy date column for plotting
        start_date = datetime(1901,11,1)
        date_list = [(start_date+timedelta(days=x)).strftime('%m-%d') for x in range(len(df))]

        df['date'] = date_list
        
        fig, lines = plot_all(fig, df, start_year, end_year)

        fig, selected_2, label_2, hover_2 = plot_selected(fig, df, end_year-1, SELECTED_COLOUR_2)
        fig, selected_1, label_1, hover_1 = plot_selected(fig, df, end_year, SELECTED_COLOUR_1)

        # Copied from below, could likely get this to be one calleable function
        # this should generally ensure that labels are not overlapping
        ybuffer = 10
        xbuffer = 30

        # label positioning check
        # first, check if vertical is overlapping
        if (label_1.y - ybuffer) < label_2.y < (label_1.y + ybuffer):

            # if so, check if horizontal is also overlapping
            #if (labely_1 - xbuffer) < labely_2 < (labely_1 + xbuffer):
            if (label_1.x - xbuffer) < label_2.x < (label_1.x + xbuffer):

                # shift x position of label_2 further right
                label_2.x = label_2.x + xbuffer
                # find y value at new x location of label_2
                label_2.y = lakes[lake]['data'].loc[label_2.x, str(end_year-1)]

        lakes[lake]['plot'] = {'fig':fig,
                               'lines':lines,
                               'selected_1':selected_1,
                               'selected_2': selected_2,
                               'label_1': label_1,
                               'label_2' : label_2}

    curdoc().theme = 'dark_minimal'

    width = PLOT_WIDTH

    # add sliders
    slider_1 = Slider(start=start_year,
                    end=end_year,
                    value=end_year,
                    step=1,
                    title='Selected Year',
                    width=width-30,
                    bar_color=SELECTED_COLOUR_1,
                    orientation='horizontal')

    slider_2 = Slider(start=start_year,
                    end=end_year-1,
                    value=end_year-1,
                    step=1,
                    title='Selected Year',
                    width=width-30,
                    bar_color=SELECTED_COLOUR_2,
                    orientation='horizontal')

    # callback to use for both sliders
    def callback(atrr, old, new):
        """
        This callback is used for both sliders, called whenever either slider
        is adjusted.  This means the entire function is called each time, so
        even if only one slider is changed, the function calls related to the
        other slider are re-run as well, which could likely be modified /
        improved; however, the benefit is that I can reassess and reposition
        over-lapping labels

        """
        for lake in lakes.keys():
            p = lakes[lake]['plot']

            p['selected_1'].glyph.name=str(slider_1.value)
            p['selected_1'].glyph.y=str(slider_1.value)
            y_1 = str(slider_1.value)
            p['fig'].tools[0].tooltips = get_tooltips(y_1)
            p['fig'].tools[0].names = [y_1]

            # find location of peak value for labelling
            labelx_1 = lakes[lake]['data'][y_1].idxmax(axis=0) # index of max ice cover
            labely_1 = lakes[lake]['data'].loc[:, y_1].max() # max ice cover

            # add label for selected year, locate at peak
            lakes[lake]['plot']['label_1'].x=labelx_1
            lakes[lake]['plot']['label_1'].y=labely_1
            lakes[lake]['plot']['label_1'].text=y_1
            lakes[lake]['plot']['label_1'].text_font_style = 'bold'

            # selected 2
            p['selected_2'].glyph.name=str(slider_2.value)
            p['selected_2'].glyph.y=str(slider_2.value)
            y_2 = str(slider_2.value)
            p['fig'].tools[0].tooltips = get_tooltips(y_2)
            p['fig'].tools[0].names = [y_2]

            # find location of peak value for labelling
            labelx_2 = lakes[lake]['data'][y_2].idxmax(axis=0) # index of max ice cover
            labely_2 = lakes[lake]['data'].loc[:, y_2].max() # max ice cover

            # this should generally ensure that labels are not overlapping
            ybuffer = 10
            xbuffer = 30

            # label positioning check
            # first, check if vertical is overlapping
            if (labely_1 - ybuffer) < labely_2 < (labely_1 + ybuffer):

                # if so, check if horizontal is also overlapping
                #if (labely_1 - xbuffer) < labely_2 < (labely_1 + xbuffer):
                if (labelx_1 - xbuffer) < labelx_2 < (labelx_1 + xbuffer):

                    # shift x position of label_2 further right
                    labelx_2 = labelx_2 + xbuffer
                    # find y value at new x location of label_2
                    labely_2 = lakes[lake]['data'].loc[labelx_2, y_2]

            # add label for selected year, locate at peak
            lakes[lake]['plot']['label_2'].x=labelx_2
            lakes[lake]['plot']['label_2'].y=labely_2
            lakes[lake]['plot']['label_2'].text=y_2
            lakes[lake]['plot']['label_2'].text_font_style = 'bold'

    # value_throttled prevents callback until slider is released, prevents lag
    slider_1.on_change('value_throttled', callback)

    slider_2.on_change('value_throttled', callback)


    plots = [None]*6
    plots[0] = lakes['Lake Superior']['plot']['fig']
    plots[1] = lakes['Lake Michigan']['plot']['fig']
    plots[2] = lakes['Lake Huron']['plot']['fig']
    plots[3] = lakes['Lake Erie']['plot']['fig']
    plots[4] = lakes['Lake Ontario']['plot']['fig']
    plots[5] = lakes['Great Lakes']['plot']['fig']


    for lake in [1,2,4,5]:
        plots[lake].yaxis.axis_label = None
        plots[lake].plot_width = PLOT_WIDTH - 25

    curdoc().title = "Great Lakes Ice Coverage"

    text_colour = ("#FFFFFF" if FULL_APP is True else 'black')
    updated="""(Data updated through {} UTC)""".format(last_data_updated.strftime('%Y-%m-%d'))

    if FULL_APP is False:

        # if not a full bokeh app, create standalone version with a title and
        # some additional descriptive text/credit info
        title = 'Great Lakes Ice Cover Concentration (1973-{})'.format(end_year)

        credits_text_1 = '''Data: NOAA Great Lakes Environmental Research Laboratory'''
        credits_text_2 = '''Graphic: Jacob Bruxer'''

        subtitle = '<br>' + credits_text_1.format(end_year) + \
                   '<br>URL : https://www.glerl.noaa.gov/data/ice/' + \
                   '<br><br>' + credits_text_2 + \
                   '<br><br>' + updated


        title= Div(text=title.format(end_year),
               style={'font-size': '150%', 'color': text_colour},
               width=width)

        subtitle=Div(text=subtitle,
                 style={'font-size': '70%', 'color': text_colour},
                 width=width)

        app_layout = layout([[title],
                            [subtitle, column(slider_1, slider_2)],
                            [plots[0], plots[1], plots[2]],
                            [plots[3], plots[4], plots[5]]])

        from bokeh.io import output_file

        output_file("out.html")
        show(app_layout)

    else:

        updated = Div(text=updated)

        # lay out components in rows/columns
        app_layout = layout([slider_1],
                            [slider_2], [updated],
                            [plots[0], plots[1], plots[2]],
                            [plots[3], plots[4], plots[5]])

        curdoc().add_root(app_layout)

    return app_layout

app_layout = build_layout()



