import h3
import pandas as pd

import json
import geopandas as gpd

import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Polygon, LineString

def part_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

def add_temporal_features(df, datetime_col):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['part_of_day'] = df[datetime_col].dt.hour.apply(part_of_day)
    return df

def lat_lng_to_h3(row, resolution=8):
    return ( 
            h3.geo_to_h3(row['source_latitude'], row['source_longitude'], resolution), 
            h3.geo_to_h3(row['destination_latitude'], row['destination_longitude'], resolution)
           )

def create_linestring(row):
    source_coords = h3.h3_to_geo(row['source_h3'])
    destination_coords = h3.h3_to_geo(row['destination_h3'])
    return LineString([(source_coords[1], source_coords[0]), (destination_coords[1], destination_coords[0])])

def create_route_map(gdf, map_style="carto-positron"):

    lon = []
    lat = []
    start_lon = []
    start_lat = []
    end_lon = []
    end_lat = []
    
    for geom in gdf.geometry:
        lon.append([coord[0] for coord in geom.coords])
        lat.append([coord[1] for coord in geom.coords])
        start_lon.append(geom.coords[0][0])
        start_lat.append(geom.coords[0][1])
        end_lon.append(geom.coords[-1][0])
        end_lat.append(geom.coords[-1][1])
    
    fig = go.Figure()

    for i in range(len(gdf)):
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=lon[i],
            lat=lat[i],
            line=dict(width=2, color='blue'),
            opacity=0.6,
            name=f"Route {i + 1}"
        ))

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=start_lon,
        lat=start_lat,
        marker=dict(size=8, color="green"),
        name="Start Points"
    ))
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=end_lon,
        lat=end_lat,
        marker=dict(size=8, color="red"),
        name="End Points"
    ))

    fig.update_layout(
        mapbox_style=map_style,
        mapbox_center={"lat": 51.50284, "lon": -0.157160},
        mapbox_zoom=12,
        margin={"r":0,"t":0,"l":0,"b":0},
        title="Most Common Routes in London"
    )

    fig.show()