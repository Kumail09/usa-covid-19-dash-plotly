import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import time
from urllib.request import urlopen
import json

#-----------------------------------------------------------------------------------------------------------------------------------------#
#Importing Data
df_juri = pd.read_csv('COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv', parse_dates=['Date'])
df_county = pd.read_csv('COVID-19_Vaccinations_in_the_United_States_County.csv', parse_dates=['Date'])
df_trans = pd.read_csv('United_States_COVID-19_County_Level_of_Community_Transmission_as_Originally_Posted.csv', parse_dates=['report_date'])

#Defining a range of Dates in data
date_range = pd.date_range(df_juri['Date'].min(), df_juri['Date'].max(), freq='D')

#Importing GeoJson for FIPS Codes - USA County
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties_j = json.load(response)


#-----------------------------------------------------------------------------------------------------------------------------------------#
#Supporting Functions
def listToDict(lt):
    return [{'label': k, 'value': k} for k in lt]

def unixTimeMillis(dt):
    return int(time.mktime(dt.timetuple()))

def unixToDatetime(unix):
    return pd.to_datetime(unix,unit='s')
#-----------------------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------------------------#
#Declaring App with Themes
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.config.suppress_callback_exceptions = True
#-----------------------------------------------------------------------------------------------------------------------------------------#
"""
App Layout for 7 Inputs and 4 Graphs
"""
app.layout = dbc.Container(children=[
    dbc.Row(dbc.Col(html.H2('US COVID-19 DATA TRACKER', className='text-center text-primary, mb-3'))),
    dbc.Row(dbc.Col([
        dcc.RadioItems(
            id='input_1',
            options=[
                {'label': 'Fully Vaccinated', 'value': 'Series_Complete_Pop_Pct'},
                {'label': 'At least 1 dose', 'value': 'Administered_Dose1_Pop_Pct'}],
            value='Administered_Dose1_Pop_Pct',
            labelStyle={'display': 'inline-block'}),
        dcc.Graph(id='graph_1', figure={}),
        dcc.Slider(
            id='input_2',
            min=unixTimeMillis(date_range.min()),
            max=unixTimeMillis(date_range.max()),
            value=unixTimeMillis(date_range.min())+86400,
        ),
        html.Div(id='slider1-output-container'),
        html.Hr()
    ])),
    dbc.Row([dbc.Col([
        dcc.Dropdown(
            id='input_3',
            options=listToDict(df_county.Recip_State.unique().tolist()),
            multi=False,
            value='GA'
        ),
        dcc.RadioItems(
            id='input_5',
            options=[
                {'label': 'Fully Vaccinated', 'value': 'Series_Complete_Pop_Pct'},
                {'label': 'At   least 1 dose', 'value': 'Administered_Dose1_Pop_Pct'}],
            value='Administered_Dose1_Pop_Pct',
            labelStyle={'display': 'inline-block'}),
        dcc.Graph(id='graph_2', figure={}),
        dcc.Slider(
            id='input_6',
            value=unixTimeMillis(date_range.min())+86400,

        ),
        html.Div(id='slider2-output-container')

    ], width={'size': 5, 'offset': 0, 'order': 1}),
    dbc.Col([
        dcc.Dropdown(
            id='input_4',
            options=[],
            multi=False
        ),
        dcc.RangeSlider(
            id='input_7',
            count=1,
            # tooltip={"placement": "bottom", "always_visible": False}

        ),
        dcc.Graph(id='graph3', style={'display': 'inline-block'}),
        dcc.Graph(id='graph4', style={'display': 'inline-block'}),
], width={'size': 5, 'offset': 0, 'order': 2})])

])
#-----------------------------------------------------------------------------------------------------------------------------------------#
#Tooltips for Slider
@app.callback(
    Output(component_id='slider1-output-container', component_property='children'),
    Input(component_id='input_2', component_property='value')
)
def input_1_tooltip(value):
    if value:
        return "Date: {}".format(unixToDatetime(value).strftime('%Y-%m-%d'))
    else:
        return "Date: {}"

@app.callback(
    Output(component_id='slider2-output-container', component_property='children'),
    Input(component_id='input_6', component_property='value')
)
def input_1_tooltip(value):
    if value:
        return "Date: {}".format(unixToDatetime(value).strftime('%Y-%m-%d'))
    else:
        return "Date: {}"


#-----------------------------------------------------------------------------------------------------------------------------------------#

#Graph 1
@app.callback(
    Output(component_id='graph_1', component_property='figure'),
    Input(component_id='input_1', component_property='value'),
     Input(component_id='input_2', component_property='value')
)
def set_graph1(vaccine_type, dt_val):
    date = unixToDatetime(dt_val)
    end_date = date.strftime('%Y-%m-%d')
    """
    Graph 1:
    Columns:
    {Dose1: Administered_Dose1_Pop_Pct
    Fully: Series_Complete_Pop_Pct
    Date
    Location}

    Input 1: Radio: {Fully Vaccinated, Atleast 1 Dose}
    Input 2: Slider: Date
    """

    dff = df_juri.copy()[['Date', 'Location', vaccine_type]]
    dff = dff[dff.Date == end_date]

    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='Location',
        scope='usa',
        color= f'{vaccine_type}',
        hover_data= [f'{vaccine_type}'],
        color_continuous_scale=px.colors.sequential.YlGnBu,
        labels={f'vaccine_type':
                'vaccinated'},
        title="U.S State Map Colored by Vaccination %"
    )
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=11,
            color="RebeccaPurple"
        ))
    return fig
#-----------------------------------------------------------------------------------------------------------------------------------------#

#Graph 2
@app.callback(
    Output('input_6', 'min'),
    Output('input_6', 'max'),
    Input('input_3', 'value')
)
def slider_2_vals(state):
    dff = df_county.copy()[['Date', 'Recip_State']]
    dff = dff[dff['Recip_State'] == state]
    try:
        dr = pd.date_range(dff['Date'].min(), dff['Date'].max(), freq='D')
    except:
        dr = pd.date_range('2020', '2021', freq='D')

    return unixTimeMillis(dr.min()), unixTimeMillis(dr.max())


@app.callback(
    [Output(component_id='graph_2', component_property='figure')],
    [Input(component_id='input_5', component_property='value'),
     Input(component_id='input_3', component_property='value'),
     Input(component_id='input_6', component_property='value')]
)
def set_graph2(vaccine_type, state, dt_val):
    date = unixToDatetime(dt_val)
    if date is not None:
        end_date = date.strftime('%Y-%m-%d')
    else:
        end_date = '2020-13-12'

    """
    Graph 2:
    Columns:
    {Dose1: Administered_Dose1_Pop_Pct
    Fully: Series_Complete_Pop_Pct
    Date
    Location}

    Input 1: Radio: {Fully Vaccinated, Atleast 1 Dose}
    Input 2: Slider: Date
    """
    dff = df_county.copy()[['Date', 'FIPS', 'Recip_State', vaccine_type]]
    dff = dff[dff['Recip_State'] == state]
    dff = dff[dff.Date == end_date]
    fig = px.choropleth(
        data_frame=dff,
        geojson=counties_j,
        locations='FIPS',
        scope='usa',
        color=f'{vaccine_type}',
        hover_data=[f'{vaccine_type}'],
        color_continuous_scale=px.colors.sequential.Brwnyl,
        labels={f'vaccine_type':
                    'vaccinated'},
        title="Selected State Colored by Vaccination %"
    )
    fig.update_geos(fitbounds='locations', visible=False)
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=11,
            color="RebeccaPurple"
        ))

    return (fig,)


#-----------------------------------------------------------------------------------------------------------------------------------------#

#Graph 3 and Input Configurations
@app.callback(
    Output('input_4', 'options'),
    Input('input_3', 'value')
)
def counties_in_state(state):
    lst = []
    dff = df_county.copy()
    dff = dff[['Recip_State', 'Recip_County', 'FIPS']]

    dff = dff[dff['Recip_State'] == state].drop_duplicates()
    if dff is not None:
        for rs, rc, f in dff.values.tolist():
            lst.append({'label': rc, 'value':f})
    return lst

@app.callback(
    Output('input_7', 'min'),
    Output('input_7', 'max'),
    Input('input_4', 'value')
)
def input_7_set(county):
    mi = 0
    mx = 1
    if county and county != 'UNK':
        dff = df_trans.copy()
        dff = dff[['report_date', 'fips_code']]
        dff = dff[dff['fips_code'] == int(county)]
        dr = pd.date_range(dff['report_date'].min(), dff['report_date'].max(), freq='D')
        mi = unixTimeMillis(dr.min())
        mx = unixTimeMillis(dr.max())
    return mi, mx

@app.callback(
    Output('graph3', 'figure'),
    Input('input_7', 'value'),
    Input('input_4', 'value'),
    Input('input_3', 'value')
)
def graph3(dr, county, state):
    dff = df_trans.copy()
    fig = {}
    if dr and county and county != 'UNK':
        dff = dff[dff['fips_code'] == int(county)]
        dff = dff[dff['report_date'] > unixToDatetime(dr[0])]
        dff = dff[dff['report_date'] < unixToDatetime(dr[1])]
        ydf = dff[['report_date', 'percent_test_results_reported_positive_last_7_days']].groupby('report_date').mean()
        x=ydf.index
        y = ydf['percent_test_results_reported_positive_last_7_days'].values
        fig = px.line(x=x, y=y,
                      title="Daily % Positivity - 7 Day Moving Average"
                      )
        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Average Percentage Daily Positive",
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=11,
                color="RebeccaPurple"
            )
        )

    return fig

#--------------------------------------------------------------------------------------------------------------------------------------------------#
#Graph 4

@app.callback(
    Output('graph4', 'figure'),
    Input('input_7', 'value'),
    Input('input_4', 'value'),
    Input('input_3', 'value')
)
def graph4(dr, county, state):
    dff = df_trans.copy()
    fig = {}
    if dr and county and county != 'UNK':
        dff = dff[dff['fips_code'] == int(county)]
        dff = dff[dff['report_date'] > unixToDatetime(dr[0])]
        dff = dff[dff['report_date'] < unixToDatetime(dr[1])]
        dff = dff[pd.to_numeric(dff['cases_per_100K_7_day_count_change'], errors='coerce').notnull()]
        dff['cases_per_100K_7_day_count_change'] = dff['cases_per_100K_7_day_count_change'].apply(pd.to_numeric)
        ydf = dff[['report_date', 'cases_per_100K_7_day_count_change']].groupby('report_date').mean()
        x = ydf.index
        y = ydf['cases_per_100K_7_day_count_change'].values
        fig = px.line(x=x, y=y,
                      title="Daily New Cases - 7 Day Moving Average per 100k"
                      )
        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Average Cases Per 100K",
            legend_title="Legend Title",
            font=dict(
                family="Cambria",
                size=11,
                color="RebeccaPurple"
            )
        )
    return fig
#--------------------------------------------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    app.run_server(debug=True)