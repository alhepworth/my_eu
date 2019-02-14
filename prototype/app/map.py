import os

from flask import Flask
from flask import render_template

import pandas as pd
from pandas import DataFrame

API_KEY = os.environ.get('MY_EU_API_KEY', None)

app = Flask(__name__)


def get_weights(beneficiaries: DataFrame) -> DataFrame:
    """
    Calculate map weights based on relative EU investment size.

    Google Maps says 3 is the default size, which is quite small.
    """

    beneficiaries = beneficiaries.copy()
    beneficiaries['weight'] = 5 * beneficiaries.eu_investment / beneficiaries.eu_investment.median()
    return beneficiaries


def get_beneficiaries() -> DataFrame:
    beneficiaries = pd.read_pickle('../data/beneficiaries.pkl')
    beneficiaries = get_weights(beneficiaries)
    return beneficiaries


def get_local_beneficiaries(map_lat: float, map_lng: float, map_zoom: int) -> DataFrame:
    """Return only projects that are fairly close to the map's centre."""

    return beneficiaries[
        (map_lat - 100 / map_zoom < beneficiaries.lat) &
        (beneficiaries.lat < map_lat + 100 / map_zoom) &
        (map_lng - 100 / map_zoom < beneficiaries.lng) &
        (beneficiaries.lng < map_lng + 100 / map_zoom)
    ][:500]


@app.route('/')
def map(map_lat=53.5, map_lng=-1.5, map_zoom=9):
    local_beneficiaries = get_local_beneficiaries(map_lat, map_lng, map_zoom)

    return render_template(
        'index.html',
        map_lat=map_lat,
        map_lng=map_lng,
        map_zoom=map_zoom,
        api_key=API_KEY,
        beneficiaries=local_beneficiaries
    )

@app.route('/about')
def about():
    return render_template('about.html')


beneficiaries = get_beneficiaries()
