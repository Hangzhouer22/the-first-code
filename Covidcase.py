#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:12:41 2022

@author: yutaoyan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as spi
import pylab as pl
countries = {
    'Korea, South': {'pop': 51.47e6},
    'Italy': {'pop': 60.48e6}, 
    'Iran': {'pop': 81.16e6}, 
    'US': {'pop': 327.2e6},
    'Spain': {'pop': 46.66e6},
    'France': {'pop': 66.99e6},
}
dfs = []
for country_name in countries.keys():
    print(country_name)
    states = ['recovered', 'confirmed', 'deaths']
    for state in states:
        df = pd.read_csv('data/{}.csv'.format(state))
        h = df[df['Country/Region'] == country_name]
        h = h.transpose().iloc[4:]
        h = h.sum(axis=1)
        h.columns = [state]
        dfs.append(h)
    country = pd.DataFrame(index=dfs[0].index)
    for df in dfs:
        country[df.columns[0]] = df

    # adding dates so that models can start earlier
    added = pd.DataFrame(0, index=['1/{}/20'.format(i) for i in range(1, 22)], columns=states)
    country = added.append(country)

    country.index = pd.to_datetime(country.index)

    plt.plot(country)
    plt.show()
    
    countries[country_name]['df'] = country
