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

from country_list import countries_for_language
import numpy as np
from countryinfo import CountryInfo

from sodapy import Socrata
import unidecode



spa = dict(countries_for_language("es"))
for k,v in spa.items():
    spa.update({k: unidecode.unidecode(v.upper())})
    

eng = dict(countries_for_language("en"))

def load_dataset(tipo):
    if tipo == "Confirmed":
        file = "time_series_covid19_confirmed_global.csv"
    elif tipo == "Deaths":
        file = "time_series_covid19_deaths_global.csv"
    else:
        file = "time_series_covid19_recovered_global.csv"

    url_confirmed=f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/{file}"
    s=requests.get(url_confirmed).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    df = df.drop(["Province/State", "Lat", "Long"], axis=1)
    df = pd.DataFrame(df.set_index("Country/Region").stack()).reset_index()
    df = df.rename(columns={"Country/Region":"Country","level_1":"Date", 0:tipo})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.groupby(["Date", "Country"])[tipo].max().reset_index()
    return df


def load_colombia_df():
    client = Socrata("www.datos.gov.co", None)  # https://www.datos.gov.co/es/profile/edit/developer_settings   por si no funciona
    results = client.get("gt2j-8ykr", limit=100000)
    results_df = pd.DataFrame.from_records(results)
    return results_df







def generar_fuera_china(df_data):
    cuenta = df_data.groupby("Country")["Confirmed", "Recovered", "Deaths"].max().reset_index()
    cuenta = cuenta[cuenta["Confirmed"]>0]
    row_latest = cuenta[((cuenta["Country"] != "China") & (cuenta["Country"] != "Others"))]
    rl = row_latest.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].max().reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
    rl.head()
    rl.head().style.background_gradient(cmap='rainbow')

    ncl = rl.copy()
    ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths'] - ncl['Recovered']
    ncl = ncl.melt(id_vars="Country", value_vars=['Affected', 'Recovered', 'Deaths'])


    return ncl

def generar_barras_fuera_china(df_data):
    ncl = generar_fuera_china(df_data)
    fig = px.bar(ncl.sort_values(['variable', 'value']), 
                y="Country", x="value", color='variable', orientation='h', height=800)
    

    fig.update_layout(title='Number of Cases outside China',
                    font=dict(
                            family="Courier New, monospace",
                            size=15,
                            color="#7f7f7f"))
    return fig


def generar_serie_tiempo(df_all):
    gdf = df_all.groupby(['Date', 'Country'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index()
    formated_gdf = gdf.copy()
    formated_gdf = formated_gdf[((formated_gdf['Country']!='China') & (formated_gdf['Country']!='Others'))]
    formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
    # formated_gdf = formated_gdf[formated_gdf["Date"] > pd.Timestamp(2020,2,1)]
    formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
    formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3).fillna(0)
    return formated_gdf

def generar_serie_tiempo_mapa(df_all):
    formated_gdf = generar_serie_tiempo(df_all)
    fig = px.scatter_geo(formated_gdf, locations="Country", locationmode='country names', 
                        color="Confirmed", size='size', hover_name="Country", 
                        projection="natural earth", animation_frame="Date", 
                        range_color= [0, max(formated_gdf['Confirmed'])+2])

    fig.update_layout(title='Spread outside China over time',
                        font=dict(
                                family="Courier New, monospace",
                                size=15,
                                color="#7f7f7f"
                            ),
                 )
        
    return fig


def generar_casos(tipo, df_data):
    importantes = ["Colombia", "Ecuador", "Italy", "Spain", "France", "US", "Germany", "South Korea", "Brazil", "Mexico", "Chile" , "Panama", "Peru"]
    df_prin = df_data[df_data["Country"].isin(importantes)]
    df_prin = df_prin[df_prin["Confirmed"] > 0]
    df_prin['start_date'] = df_prin.groupby('Country')['Date'].transform('min')
    df_prin["days_since_start"] = (df_prin["Date"] - df_prin['start_date']).dt.days
    df_prin = df_prin.replace("US", "United States")
    return df_prin

def generar_casos_graf(tipo, df_data):
    df_prin = generar_casos(tipo, df_data)
    fig = px.line(df_prin, x="days_since_start", y=tipo, color='Country')

    fig.update_layout(title='CoVid cases',
                   xaxis_title='Days from first confirmed in each country',
                   yaxis_title=f'# of {tipo}',
                   xaxis=dict(range=[0, 60]),
                   yaxis=dict(range=[0, 5000]),
                   font=dict(
                        family="Courier New, monospace",
                        size=15,
                        color="#7f7f7f"
                    ),
                 )
    return fig


def generar_casos_porcentaje(df_data):
    df_prin = generar_casos("Deaths", df_data)
    unicos = df_prin["Country"].unique()
    pops = [CountryInfo(p).info()["population"] for p in unicos]
    df_pops = pd.DataFrame({"Country":unicos, "population":pops})

    df_prin = df_prin.merge(df_pops, on="Country", how="left")    
    df_prin["percentage"] = df_prin["Deaths"]/df_prin["population"]
    return df_prin

def generar_casos_porcentaje_graf(df_data):
    df_prin = generar_casos_porcentaje(df_data)
    fig = px.line(df_prin, x="days_since_start", y="percentage", color='Country')

    fig.update_layout(title='CoVid cases percentages by country',
                   xaxis_title='Days from first confirmed in each country',
                   yaxis_title=f'% of Confirmed',

                   font=dict(
                        family="Courier New, monospace",
                        size=15,
                        color="#7f7f7f"
                    ),
                 )
    return fig


def generar_cuenta_colombia(df_data):
    lista = ["Amazonas","Antioquia","Arauca","Atlántico","Bogotá D.C.","Bolívar",
    "Boyacá","Caldas","Caquetá","Casanare","Cauca","Cesar","Chocó",
    "Córdoba","Cundinamarca","Guainía","Guaviare","Huila","La Guajira","Magdalena",
    "Meta","Nariño","Norte de Santander","Putumayo","Quindío","Risaralda","San Andrés y Providencia",
    "Santander","Sucre","Tolima","Valle del Cauca","Vaupés","Vichada"]
    lista = [depto.upper() for depto in lista]
    ceros = [0]*len(lista)
    df_ceros = pd.DataFrame({"NOMBRE_DPT":lista, "cuenta_ceros":ceros})

    # df_data = pd.read_csv("data/Casos1.csv")
    df_data = df_data.rename(columns={"departamento": "NOMBRE_DPT"})
    df_data["NOMBRE_DPT"] = df_data["NOMBRE_DPT"].str.upper()
    df_cuenta = pd.DataFrame(df_data.groupby("NOMBRE_DPT")["id_de_caso"].count()).reset_index().rename(columns={"id_de_caso": "cuenta"})
    
    df_merge = df_ceros.merge(df_cuenta, on="NOMBRE_DPT", how="left")


    df_merge["total"] = df_merge["cuenta"] + df_merge["cuenta_ceros"]
    df_merge = df_merge.drop(["cuenta", "cuenta_ceros"], axis=1)
    nombres_dict = {"BOGOTÁ D.C.": "SANTAFE DE BOGOTA D.C",
                    "VALLE": "VALLE DEL CAUCA"}
    for dept in nombres_dict:
        df_merge = df_merge.replace(dept, nombres_dict[dept])
    df_merge = df_merge.fillna(0)
    df_merge['NOMBRE_DPT'] = df_merge['NOMBRE_DPT'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_merge = df_merge.replace("NARINO", "NARIÑO")
    return df_merge

def generar_mapa_colombia_cuenta(df_data):
    df_merge = generar_cuenta_colombia(df_data)
    with urlopen('https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json') as response:
        counties = json.load(response)

    fig = px.choropleth(df_merge, geojson=counties, color="total",
                        locations="NOMBRE_DPT", featureidkey="properties.NOMBRE_DPT",
                        projection="mercator",scope="south america",color_continuous_scale=px.colors.sequential.Blues
                    )
    fig.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="Black",
        showsubunits=True)
    fig.update_layout(
        title_text = 'Confirmados en Colombia',
        font=dict(
            family="Courier New, monospace",
            size=25,
            color="#7f7f7f"
        )
    )
    return fig


def arreglar_fecha(x):
    if len(x) > 13:
        lista = x.split("-")
        return f"{lista[1]}/{lista[2][:2]}/{lista[0]}"
    else:
        lista = x.split("/")
        if len(lista[1]) == 1:
            mes = "0" + lista[1]
            return f"{lista[0]}/{mes}/{lista[2]}"
        elif len(lista[1]) == 2:
            return x


def generar_por_dia_colombia(df_data):
    # df_data = pd.read_csv("data/Casos1.csv")
    df_data = df_data.rename(columns={"fecha_de_diagn_stico": "Fecha de diagnóstico"})
    df_data["Fecha de diagnóstico"] = df_data["Fecha de diagnóstico"].apply(arreglar_fecha)
    df_data["Fecha de diagnóstico"] = pd.to_datetime(df_data["Fecha de diagnóstico"], format = "%d/%m/%Y", errors="coerce")
    cuenta = pd.DataFrame(df_data.groupby("Fecha de diagnóstico")["id_de_caso"].count()).reset_index()
    cuenta = cuenta.rename(columns={"id_de_caso":"cuenta"})
    return cuenta

def generar_por_dia_barras_colombia(df_data):
    cuenta = generar_por_dia_colombia(df_data)
    fig = px.bar(data_frame=cuenta, x="Fecha de diagnóstico", y="cuenta")
    fig.update_layout(
        title_text = 'Confirmados por día en Colombia',
        font=dict(
            family="Courier New, monospace",
            size=10,
            color="#7f7f7f"
        ),
        titlefont= dict(size= 25)

    )
    return fig


def get_code(row):
    try:
        indice = list(spa.values()).index(row["País de procedencia"])
        codigo = list(spa.keys())[indice]
        return codigo
    except:
        # print(row)
        return "CO"

def get_lat_long(row, df_lat_lon):
    try:
        cod =row["codigos"]
        row_cod = df_lat_lon[df_lat_lon["code"] == cod]
        row["lat"] = row_cod["lat"].values[0]
        row["long"] = row_cod["lon"].values[0]
    except:
        print(row_cod)
        print(row)
    return row

def generar_cuenta_importados(df_data):
    df_lat_lon = pd.read_csv("data/lat_long.csv")
    # df_data = pd.read_csv("data/Casos1.csv")
    df_data = df_data.rename(columns={"pa_s_de_procedencia": "País de procedencia"})
    df_data = df_data.loc[:,["sexo", "País de procedencia"]]
    df_data["País de procedencia"] = df_data["País de procedencia"].fillna("Colombia")
    df_data["País de procedencia"] = df_data["País de procedencia"].apply(lambda x: x.split("-")[0])
    df_data["País de procedencia"] = df_data["País de procedencia"].str.strip()
    df_data["País de procedencia"] = df_data["País de procedencia"].str.upper()
    df_data = df_data.replace("ESTADOS UNIDOS DE AMÉRICA", "ESTADOS UNIDOS")
    df_data = df_data.replace("ESPAÑA", "ESPANA")
    df_data = df_data.replace("Isla Martín", "Colombia")
    df_data["codigos"] = df_data.apply(get_code, axis=1)
    df_data["name"] = df_data.apply(lambda x: eng[x["codigos"]], axis=1)
    paises = df_data.apply(get_lat_long, df_lat_lon=df_lat_lon, axis=1).loc[:,["name", "lat", "long"]]
    paises = paises.drop_duplicates()
    df_data = df_data.merge(paises, on="name", how="left")
    df_data["end_lat"] = 2.889443
    df_data["end_long"] = -73.783892
    df_data_grouped = df_data.groupby(["name", "lat", "long", "end_lat", "end_long"]).count().reset_index().drop(["País de procedencia", "codigos"], axis=1)
    df_data_grouped = df_data_grouped.rename(columns={"sexo":"cuenta"})
    df_data_grouped["texto"] = df_data_grouped["name"] + ", number of confirmed: " + df_data_grouped["cuenta"].astype(str)
    return df_data_grouped

def generar_mapa_importados(df_data):
    df_data_grouped = generar_cuenta_importados(df_data)
    fig = go.Figure()
    for i in range(len(df_data_grouped)):
        fig.add_trace(
            go.Scattergeo(
                lon = [df_data_grouped['long'][i], df_data_grouped['end_long'][i]],
                lat = [df_data_grouped['lat'][i], df_data_grouped['end_lat'][i]],
                mode = 'lines',
                line = dict(width = 2*(np.log(df_data_grouped['cuenta'][i]) / np.log(df_data_grouped['cuenta'].max())),
                            color = '#051C57'),
                opacity = 1,
            )
        )

    fig.add_trace(go.Scattergeo(
        lon = df_data_grouped['long'],
        lat = df_data_grouped['lat'],
        text = df_data_grouped['texto'],
        hoverinfo = "text",
        mode = 'markers',
        marker = dict(
            size = 10,
            opacity = (np.log(df_data_grouped['cuenta']) / np.log(df_data_grouped['cuenta'].max())),
            color="#051C57"
        )))

    fig.update_layout(
        title_text = 'Procedencia de confirmados',
        showlegend = False,
        font=dict(
            family="Courier New, monospace",
            size=28,
            color="#7f7f7f"
        ),
        geo = dict(
            scope = 'world',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
        ),
    )
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



df_col = load_colombia_df()
df_col["id_de_caso"] = df_col["id_de_caso"].astype(int)
total_colombia = df_col["id_de_caso"].max()





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


app.layout = dbc.Container([
    html.Div([
        html.H2("Coronavirus", className="pretty_container", style={'text-align': 'center'}),
        html.Div([html.H2("Confirmed: ", style = {"color": "#14378F "}),
                  html.H2(total_confirmed)], 
                className="pretty_container", style={'text-align': 'center'}),

        html.Div([html.H2("Deaths: ", style = {"color": "#AF1E3E "}),
                        html.H2(total_deaths)], 
                        className="pretty_container", style={'text-align': 'center'}),

        html.Div([html.H2("Recovered: ", style = {"color": "#07830D"}),
                  html.H2(total_recovered)], 
                className="pretty_container", style={'text-align': 'center'}),
        
        html.Div([html.H2("Confirmados en Colombia: ", style = {"color": "#89690E"}),
                  html.H2(total_colombia)], 
                className="pretty_container", style={'text-align': 'center'}),
        ],className="pretty_container"

    ),
    
    html.Div([  dbc.Row([
                    html.Div(
                        dcc.Graph(
                            id = "mapa_colombia",
                            figure = generar_mapa_colombia_cuenta(df_col)
                        ), className="pretty_container"
                    ),
                ]),
                dbc.Row([
                    html.Div(
                        dcc.Graph(
                            id = "mapa_colombia_importados",
                            figure = generar_mapa_importados(df_col)
                        ), className="pretty_container"
                    ),
                ]),
                dbc.Row([
                    html.Div(
                        dcc.Graph(
                            id = "barras_por_dia_colombia",
                            figure = generar_por_dia_barras_colombia(df_col)
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
                            id = "casos_porcentaje",
                            figure = generar_casos_porcentaje_graf(df_data)
                        ),className="pretty_container"
                    )
                ),
                

                dbc.Row(
                    html.Div(
                        dcc.Graph(
                            id = "mapa3",
                            figure = generar_serie_tiempo_mapa(df_all)
                        ),className="pretty_container"
                    )
                ),
                

                dbc.Row(
                    html.Div(
                        dcc.Graph(
                            id = "mapa2",
                            figure = generar_barras_fuera_china(df_data)
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
        maximo = 10000
    elif input_value == "Deaths":
        maximo = 1000
    else:
        maximo = 10000
    fig = px.choropleth(cuenta, 
                            locations="Country", 
                            locationmode='country names', 
                            color=input_value, 
                            hover_name="Country", 
                            range_color=[1,maximo], 
                            color_continuous_scale="Sunsetdark")  

    fig.update_layout(title=f'Countries with {input_value} Cases',
                        font=dict(
                                family="Courier New, monospace",
                                size=15,
                                color="#7f7f7f"
                            ),
                 )
    return fig



@app.callback(Output("casos", "figure"),
            [Input('radio_mapa', 'value')])

def update_mapa2(input_value):
    fig = generar_casos_graf(input_value, df_data)
    return fig



if __name__ == "__main__":
    app.run_server(port=4070)
