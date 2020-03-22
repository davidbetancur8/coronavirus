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
import io
import requests


def load_dataset(tipo):
    url_confirmed=f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-{tipo}.csv"
    s=requests.get(url_confirmed).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    df = df.drop(["Province/State", "Lat", "Long"], axis=1)
    df = pd.DataFrame(df.set_index("Country/Region").stack()).reset_index()
    df = df.rename(columns={"Country/Region":"Country","level_1":"Date", 0:tipo})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.groupby(["Date", "Country"])[tipo].max().reset_index()
    return df

def generar_fuera_china():
    cuenta = df_data.groupby("Country")["Confirmed", "Recovered", "Deaths"].max().reset_index()
    cuenta = cuenta[cuenta["Confirmed"]>0]
    row_latest = cuenta[((cuenta["Country"] != "China") & (cuenta["Country"] != "Others"))]
    rl = row_latest.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].max().reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
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

def generar_casos(tipo):
    importantes = ["Colombia", "Italy", "Spain", "France", "US", "Germany", "South Korea"]
    df_prin = df_data[df_data["Country"].isin(importantes)]
    df_prin = df_prin[df_prin["Confirmed"] > 0]
    df_prin['start_date'] = df_prin.groupby('Country')['Date'].transform('min')
    df_prin["days_since_start"] = (df_prin["Date"] - df_prin['start_date']).dt.days

    fig = px.line(df_prin, x="days_since_start", y=tipo, color='Country')

    fig.update_layout(title='CoVid cases',
                   xaxis_title='Days from first confirmed in each country',
                   yaxis_title=f'# of {tipo}',
                   xaxis=dict(range=[0, 50]),
                   yaxis=dict(range=[0, 500])
                 )
    return fig


df_confirmed = load_dataset("Confirmed")
df_recovered = load_dataset("Recovered")
df_deaths = load_dataset("Deaths")


new_df = pd.merge(df_confirmed, df_recovered,  how='left', left_on=["Date", "Country"], right_on = ["Date", "Country"])
df_data = pd.merge(new_df, df_deaths,  how='left', left_on=["Date", "Country"], right_on = ["Date", "Country"])

df_all = df_data.copy()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


app.layout = dbc.Container([
    html.Div([
        html.H2("Coronavirus", className="pretty_container", style={'text-align': 'center'}),
        ],className="pretty_container"

    ),
    
    html.Div([
                dbc.Row([
                    html.Div(
                        dcc.RadioItems( 
                        id="radio_mapa",
                            options = [
                                {'label': 'Confirmed', 'value': "Confirmed"},
                                {'label': 'Death', 'value': "Deaths"},
                                {'label': 'Recovered', 'value': "Recovered"}
                            ],
                            value = "Confirmed"
                        ), className="pretty_container"
                    )
                    
                ]),
                dbc.Row([
                    html.Div(
                        dcc.Graph(
                            id = "mapa1",
                        ), className="pretty_container"
                    ),
                ]),
                

                dbc.Row(
                    html.Div(
                        dcc.Graph(
                            id = "casos",
                        ),className="pretty_container"
                    )
                ),
                

                dbc.Row(
                    html.Div(
                        dcc.Graph(
                            id = "mapa3",
                            figure = generar_serie_tiempo_mapa()
                        ),className="pretty_container"
                    )
                ),
                

                dbc.Row(
                    html.Div(
                        dcc.Graph(
                            id = "mapa2",
                            figure = generar_fuera_china()
                        ),className="pretty_container"
                    )
                )

    ])


], fluid=True)



@app.callback(Output("mapa1", "figure"),
            [Input('radio_mapa', 'value')])

def update_mapa1(input_value):
    cuenta = df_data.groupby("Country")[input_value].max().reset_index()
    cuenta = cuenta[cuenta[input_value]>0]
    if input_value == "Confirmed":
        maximo = 1000
    elif input_value == "Deaths":
        maximo = 50
    else:
        maximo = 100
    mapa = px.choropleth(cuenta, 
                            locations="Country", 
                            locationmode='country names', 
                            color=input_value, 
                            hover_name="Country", 
                            title=f'Countries with {input_value} Cases', 
                            range_color=[1,maximo], 
                            color_continuous_scale="Sunsetdark")   
    return mapa



@app.callback(Output("casos", "figure"),
            [Input('radio_mapa', 'value')])

def update_mapa2(input_value):
    fig = generar_casos(input_value)
    return fig



if __name__ == "__main__":
    app.run_server(port=4080)
