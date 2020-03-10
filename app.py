# ===================================================================
# Importante tener el stylesheet.css dentro de el directorio assets
# ===================================================================

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go
from dash.dependencies import Output, Input, State
import pandas as pd
import plotly.express as px



df_data = pd.read_csv("data/covid_19_data.csv")
df_data["Country"] = df_data["Country/Region"].str.replace("Mainland China", "China")
df_data["Date"] = df_data["ObservationDate"]
df_all = df_data.copy()
df_data = df_data[df_data["Date"] == max(df_data["Date"])]


def generar_fuera_china():
    cuenta = df_data.groupby("Country")["Confirmed", "Recovered", "Deaths"].sum().reset_index()
    cuenta = cuenta[cuenta["Confirmed"]>0]
    row_latest = cuenta[((cuenta["Country"] != "China") & (cuenta["Country"] != "Others"))]
    rl = row_latest.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
    rl.head()
    rl.head().style.background_gradient(cmap='rainbow')

    ncl = rl.copy()
    ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths'] - ncl['Recovered']
    ncl = ncl.melt(id_vars="Country", value_vars=['Affected', 'Recovered', 'Deaths'])

    fig = px.bar(ncl.sort_values(['variable', 'value']), 
                y="Country", x="value", color='variable', orientation='h', height=800,
                # height=600, width=1000,
                title='Number of Cases outside China')
    return fig

def generar_serie_tiempo_mapa():
    gdf = df_all.groupby(['Date', 'Country'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index()
    formated_gdf = gdf.copy()
    formated_gdf = formated_gdf[((formated_gdf['Country']!='China') & (formated_gdf['Country']!='Others'))]
    formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
    formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

    fig = px.scatter_geo(formated_gdf, locations="Country", locationmode='country names', 
                        color="Confirmed", size='Confirmed', hover_name="Country", range_color= [0, max(formated_gdf['Confirmed'])+2], 
                        projection="natural earth", animation_frame="Date", title='Spread outside China over time')
        
    return fig

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


app.layout = dbc.Container([
    html.Div([
        html.H2(children="Coronavirus"),
        ],

    ),
    

    html.Div([

        dbc.Row([
            dbc.Col([

                dbc.Row([
                    html.Div(
                        dcc.Graph(
                            id = "mapa1",
                        )
                    ),
                ]),
                dbc.Row([
                    dcc.RadioItems( 
                    id="radio_mapa",
                        options = [
                            {'label': 'Confirmed', 'value': "Confirmed"},
                            {'label': 'Death', 'value': "Deaths"},
                            {'label': 'Recovered', 'value': "Recovered"}
                        ],
                        value = "Confirmed"
                    ), 
                ]),

                dbc.Row(
                    html.Div(
                        dcc.Graph(
                            id = "mapa3",
                            figure = generar_serie_tiempo_mapa()
                        )
                    )
                )



            ]),

            dbc.Col([
                html.Div(
                    dcc.Graph(
                        id = "mapa2",
                        figure = generar_fuera_china()
                    )
                )


            ]),

        ])

    ])


], fluid=True)



@app.callback(Output("mapa1", "figure"),
            [Input('radio_mapa', 'value')])

def update_mapa1(input_value):
    cuenta = df_data.groupby("Country")["Country", input_value].sum().reset_index()
    cuenta = cuenta[cuenta[input_value]>0]
    if input_value == "Confirmed":
        maximo = 500
    elif input_value == "Deaths":
        maximo = 5
    else:
        maximo = 20
    mapa = px.choropleth(cuenta, 
                            locations="Country", 
                            locationmode='country names', 
                            color=input_value, 
                            hover_name="Country", 
                            title=f'Countries with {input_value} Cases', 
                            range_color=[1,maximo], 
                            color_continuous_scale="Sunsetdark")   
    return mapa



if __name__ == "__main__":
    app.run_server(port=4060)
