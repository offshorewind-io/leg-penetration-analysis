import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from empirical_lpa_model import calculate_grid_capacity
import plotly.io as pio

pio.templates.default = "plotly_white"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

soil_profile = pd.DataFrame(data={
    'Depth to [m]': [2, 4, 8, 10],
    'Soil type': ['Sand', 'Clay', 'Sand', 'Clay'],
    'Buoyant unit weight [kN/m3]': [9, 9, 9, 9],
    'Peak undrained shear strength [kPa]': [np.nan, 100, np.nan, 300],
    'Peak angle of friction [deg]': [35, np.nan, 40, np.nan]
})

app.layout = html.Div([
    html.H1('LPA - leg penetration assessment using the InSafe JIP mechanism based method'),
    html.P('This is a demonstration app, it should not be used in decision making processes without supervision from a qualified engineer. For consultancy services on software implementation or geotechnical analysis, please contact info@offshorewind.io'),

    dbc.Row([
        dbc.Col(
            html.Div([
                dbc.Label('Spudcan'),
                dbc.Select(id='spudcan', value='Vole au vent', options=[{"label": 'Vole au vent', 'value': 'Vole au vent'}]),
                dbc.Label('Depth increment (m)'),
                dbc.Input(id='d-z', type='number', value=0.1, disabled=True),
                dbc.Button("Calculate", id="calculate-button", ),
            ], className="d-grid gap-2"),
        md=3),
        dbc.Col(
            html.Div([
                dash_table.DataTable(soil_profile.to_dict('records'),
                    [{"name": i, "id": i} for i in soil_profile.columns],
                    id='soil-layering-table', editable=True, row_deletable=True),
                dbc.Button('Add Row', id='add-row-button', color='secondary'),
            ], className="d-grid gap-2")
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id="lpa-results-figure")
        )
    ]),
])


@app.callback(
    dash.dependencies.Output('soil-layering-table', 'data'),
    dash.dependencies.Input('add-row-button', 'n_clicks'),
    [dash.dependencies.State('soil-layering-table', 'data'),
    dash.dependencies.State('soil-layering-table', 'columns')])
def add_row(n_clicks, data, columns):
    if n_clicks is not None:
        data.append({c['id']: '' for c in columns})
    return data


@app.callback(
    dash.dependencies.Output("lpa-results-figure", "figure"),
    [dash.dependencies.Input("calculate-button", "n_clicks")],
    [dash.dependencies.State("soil-layering-table", "data"),
     dash.dependencies.State("d-z", "value"),
     dash.dependencies.State("spudcan", "value")])
def update_figure(n_clicks, soil_profile_dict, d_z, spudcan_name):

    soil_profile = pd.DataFrame(data=soil_profile_dict)
    soil_profile.index = soil_profile.index.astype(int)
    soil_profile = soil_profile.replace([''], [None], regex=True)
    soil_profile['Depth to [m]'] = soil_profile['Depth to [m]'].astype(float)
    soil_profile['Buoyant unit weight [kN/m3]'] = soil_profile['Buoyant unit weight [kN/m3]'].astype(float)
    soil_profile['Peak undrained shear strength [kPa]'] = soil_profile['Peak undrained shear strength [kPa]'].astype(float)
    soil_profile['Peak angle of friction [deg]'] = soil_profile['Peak angle of friction [deg]'].astype(float)

    if spudcan_name == 'Vole au vent':
        spudcan = pd.DataFrame(data={
            'Depth [m]': [-0.8, 0, 1],
            'Area [m2]': [0, 124.5, 124.5]
        })

    soil_profile['Depth from [m]'] = [0] + soil_profile['Depth to [m]'].tolist()[:-1]
    soil_profile['Layer thickness [m]'] = soil_profile['Depth to [m]'] - soil_profile['Depth from [m]']

    colors = {'Sand': 'sandybrown', 'Clay': 'steelblue'}
    soil_profile['Color'] = [colors[soil_type] for soil_type in soil_profile['Soil type']]
    soil_profile['Color'] = [colors[soil_type] for soil_type in soil_profile['Soil type']]

    grid = np.arange(spudcan['Depth [m]'][0], soil_profile['Depth to [m]'].tolist()[-1], d_z).tolist()
    results = calculate_grid_capacity(grid, soil_profile, spudcan)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(
        "Leg penetration assessment", "Soil profile"))

    fig.add_trace(go.Scatter(x=results["Q_v uniform [kN]"], y=results["h [m]"], name='Uniform'), 1, 1)
    fig.add_trace(go.Scatter(x=results["Q_v mechanism [kN]"], y=results["h [m]"], name='Mechanism'), 1, 1)
    fig.add_trace(go.Bar(name='Soil profile', x=[" "] * len(soil_profile), y=soil_profile["Layer thickness [m]"], marker={'color': soil_profile["Color"]}), 1, 2)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(rangemode="tozero")
    fig['layout']['yaxis']['title'] = 'Depth (m)'
    fig['layout']['xaxis']['title'] = 'Capacity (kN)'
    fig['layout']['height'] = 800

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
