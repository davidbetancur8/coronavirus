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
from urllib.request import urlopen
import json

def load_dataset(tipo):
    if tipo == "Confirmed":
        file = "time_series_covid19_confirmed_global.csv"
    elif tipo == "Deaths":
        file = "time_series_covid19_deaths_global.csv"
    else:
        file = "time_series_19-covid-Recovered.csv"

    url_confirmed=f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/{file}"
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
    # formated_gdf = formated_gdf[formated_gdf["Date"] > pd.Timestamp(2020,2,1)]
    formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
    formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

    fig = px.scatter_geo(formated_gdf, locations="Country", locationmode='country names', 
                        color="Confirmed", size='size', hover_name="Country", 
                        projection="natural earth", animation_frame="Date", 
                        range_color= [0, max(formated_gdf['Confirmed'])+2],
                        title='Spread outside China over time')
        
    return fig

def generar_casos(tipo):
    importantes = ["Colombia", "Italy", "Spain", "France", "US", "Germany", "South Korea", "Brazil", "Mexico", "Chile" , "Panama"]
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

def genrar_cuenta_colombia():
    lista = ["Amazonas","Antioquia","Arauca","Atlántico","Bogotá","Bolívar",
    "Boyacá","Caldas","Caquetá","Casanare","Cauca","Cesar","Chocó",
    "Córdoba","Cundinamarca","Guainía","Guaviare","Huila","La Guajira","Magdalena",
    "Meta","Nariño","Norte de Santander","Putumayo","Quindío","Risaralda","San Andrés y Providencia",
    "Santander","Sucre","Tolima","Valle del Cauca","Vaupés","Vichada"]
    lista = [depto.upper() for depto in lista]
    ceros = [0]*len(lista)
    df_ceros = pd.DataFrame({"NOMBRE_DPT":lista, "cuenta_ceros":ceros})

    df_data = pd.read_csv("data/Casos1.csv")
    df_data = df_data.rename(columns={"Departamento": "NOMBRE_DPT"})
    df_data["NOMBRE_DPT"] = df_data["NOMBRE_DPT"].str.upper()
    df_cuenta = pd.DataFrame(df_data.groupby("NOMBRE_DPT")["ID de caso"].count()).reset_index().rename(columns={"ID de caso": "cuenta"})
    df_merge = df_ceros.merge(df_cuenta, on="NOMBRE_DPT", how="left")
    df_merge["total"] = df_merge["cuenta"] + df_merge["cuenta_ceros"]
    df_merge = df_merge.drop(["cuenta", "cuenta_ceros"], axis=1)
    nombres_dict = {"BOGOTÁ": "SANTAFE DE BOGOTA D.C",
                    "VALLE": "VALLE DEL CAUCA"}
    for dept in nombres_dict:
        df_merge = df_merge.replace(dept, nombres_dict[dept])
    df_merge = df_merge.fillna(0)
    df_merge['NOMBRE_DPT'] = df_merge['NOMBRE_DPT'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_merge = df_merge.replace("NARINO", "NARIÑO")
    return df_merge

def generar_mapa_colombia_cuenta():
    df_merge = genrar_cuenta_colombia()
    with urlopen('https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json') as response:
        counties = json.load(response)

    fig = px.choropleth(df_merge, geojson=counties, color="total",
                        locations="NOMBRE_DPT", featureidkey="properties.NOMBRE_DPT",
                        projection="mercator",scope="south america",color_continuous_scale=px.colors.sequential.Reds
                    )
    fig.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="Black",
        showsubunits=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig




df_confirmed = load_dataset("Confirmed")
df_recovered = load_dataset("Recovered")
df_deaths = load_dataset("Deaths")


new_df = pd.merge(df_confirmed, df_recovered,  how='left', left_on=["Date", "Country"], right_on = ["Date", "Country"])
df_data = pd.merge(new_df, df_deaths,  how='left', left_on=["Date", "Country"], right_on = ["Date", "Country"])

df_all = df_data.copy()
total_confirmed = df_data.groupby("Country")["Confirmed"].max().sum()
total_deaths = df_data.groupby("Country")["Deaths"].max().sum()
total_recovered = df_data.groupby("Country")["Recovered"].max().sum()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


app.layout = dbc.Container([
    html.Div([
        html.H2("Coronavirus", className="pretty_container", style={'text-align': 'center'}),
        html.Div([html.H2("Confirmed: ", style = {"color": "blue"}),
                  html.H2(total_confirmed)], 
                className="pretty_container", style={'text-align': 'center'}),

        html.Div([html.H2("Deaths: ", style = {"color": "red"}),
                        html.H2(total_deaths)], 
                        className="pretty_container", style={'text-align': 'center'}),

        html.Div([html.H2("Recovered: ", style = {"color": "green"}),
                  html.H2(total_recovered)], 
                className="pretty_container", style={'text-align': 'center'}),
        ],className="pretty_container"

    ),
    
    html.Div([  dbc.Row([
                    html.Div(
                        dcc.Graph(
                            id = "mapa_colombia",
                            figure = generar_mapa_colombia_cuenta()
                        ), className="pretty_container"
                    ),
                ]),
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
