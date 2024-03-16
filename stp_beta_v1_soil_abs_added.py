#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:37:40 2023

@author: tariromashongamhende
"""


def black_header(url):
    st.markdown(f'<p style="background-color:#FFFFFF;color:#000000;font-size:50px:border-radius:0%;">{url}</p>', unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import os
import calplot

import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import polars as pl
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pickle
import math
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from sklearn.metrics import confusion_matrix
import itertools
import joblib
from tensorflow.keras.models import load_model






# generate written text stating when the last time there was a watering event
def get_date_plant_was_last_watered(dataframe):
    watering_events_dates = dataframe[dataframe['watering_event']==1]
    most_recent_watering_event = watering_events_dates['date'].max()
    print('your plant was watered last on '+ most_recent_watering_event.strftime("%B %d, %Y"))
    return 'Your plant was watered last on '+ most_recent_watering_event.strftime("%B %d, %Y")




plant_mapping_ids = []




st.set_page_config(layout='wide' )


logo, header = st.columns([3,10])

with logo:
    st.image('stp_logo.jpg', width=150)
    
    # directory with latest moisture readings
# =============================================================================
#     moisture_readings_directory = '/Users/tariromashongamhende/Documents/save_this_plant_beta/plant_readings/'
#     
#     plant_names_list = []
#     
#     df_container = []
#     for i in os.listdir(moisture_readings_directory):
#         if '.csv' in i:
#             plant_name = ('_'.join(i.split('-2023')[0].split('_')[1:]).replace('-','')[1:])
#             plant_names_list.append(plant_name)
#             # load csv file with moisture readings 
#             df = pd.read_csv(moisture_readings_directory+'/'+i)
#             df['plant_name'] = plant_name
#             df_container.append(df)
#     df = pd.concat(df_container)
# =============================================================================
    
    df = pd.read_parquet('2024_03_16_master_test_combined_plant_sensor_soil_moisture_readings_data_all_soil_moisture_abs_rate_added.parquet.gzip').rename(columns={'name':'plant_name'})
    df['plant_name'] = df['plant_name'].str.split(' - ').str[-1]

    plant_names_list = [x for x in df.plant_name.value_counts().index]
    
    options_dict = {'plant_names':plant_names_list}
    selected_plant = st.selectbox('Which plant do you want to check in on?', options_dict['plant_names'])

with header:
    
    # st.title(' ')
    
# sidebar, main_page = st.columns([3,7])
# with sidebar:
    


# with main_page:
    
    st.title('Save this plant mvp v1.0')
    

    df = df[df['plant_name']==selected_plant]
    
    selected_feed_id = df[df['plant_name']==selected_plant].feed_id.value_counts().index[0]
    
    df_2 = df.copy()

    # st.dataframe(df)
    
    
    # load the data
    
    # df = pd.read_csv('/Users/tariromashongamhende/Documents/save_this_plant_beta/plant_readings/test-corner-soil-moisture-3_-_Peace_Lily-20231014-1044.csv')
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    
    new_df = df[['date','soil_moisture_value','soil_moisture_absorption_rate','watering_event']]
    df = df[['date','soil_moisture_value']]
    # df.drop(columns=['id','feed_id','created_at','lat','lon','ele'], inplace=True)
        
    # st.dataframe(df)
    
    df_min = df.groupby('date').min()
    df_min.columns = ['min_value']
    df_mean = df.groupby('date').mean()
    df_mean.columns = ['mean_value']
    df_max = df.groupby('date').max()
    df_max.columns = ['max_value']
    df = df_min.merge(df_mean, left_index=True, right_index=True).merge(df_max, left_index=True, right_index=True).reset_index()
    
    df['prev_min_value'] = df['min_value'].shift(1)
    df['prev_mean_value'] = df['mean_value'].shift(1)
    df['prev_max_value'] = df['max_value'].shift(1)
    
    # st.dataframe(df)


    # it looks like if there is a big difference between the mean of a daya and max of a day it 
    # looks like that can be used to determine if there was a watering event
    df.loc[df.max_value/df.mean_value>1.5,'watering_event'] = 1
    df.loc[df.watering_event.isna(),'watering_event'] = 0
    df.loc[(df.min_value==0) &(df.mean_value!=0),'watering_event'] = 1
    df.loc[(abs(df.min_value/df.prev_min_value)>1.8)&
           (df.max_value>=df.prev_max_value*2),'watering_event'] = 1
    df.loc[(df.max_value/df.mean_value<1.5),'watering_event'] = 0
    df.loc[(df.max_value==100)&(df.prev_max_value<80),'watering_event'] = 1
    df.loc[(df.min_value>40)&(df.prev_min_value==0),'watering_event'] = 1
    # for some plants the variation between mean and max moisture readings isn't as extreme
    # for the plants we will just check that any day where the min reading == 0 then we will assume
    # that this plant was watered on this date
    
    
    


    # this will be changed to a version that is based solely on the water absorption rate
    #watering_df = df[df.watering_event==1]
    # this is the new version

    
# =============================================================================
#     if 'zz' in selected_plant:
#         print(plant_name)
#         new_df.loc[new_df.soil_moisture_absorption_rate>5, 'watering_event'] = 1
#         watering_df = new_df[new_df.watering_event==1]
#     else:
#         print(plant_name)
# 
#         new_df.loc[new_df.soil_moisture_absorption_rate>1.5, 'watering_event'] = 1


    watering_df = new_df[new_df.watering_event==1]
    
    if len(watering_df)>=1:
        date_last_watered_sentence = get_date_plant_was_last_watered(watering_df)
    
    else:
        date_last_watered_sentence = "this plant has not been watered since you've been usings sensors"
        
    # st.dataframe(watering_df)
    
    if len(watering_df)>1:
        watering_df['last_water_date'] = watering_df['date'].shift(1)
        watering_df['days_since_last_water'] = abs(pd.to_datetime(watering_df.date) - pd.to_datetime(watering_df.last_water_date)).dt.days
        
    
    
        average_number_of_days_between_waterings = watering_df.days_since_last_water.mean()
    
    print(date_last_watered_sentence)
    
    calendar, watering_date_text_box = st.columns([7,3])
    with calendar:
        
            ## Create a figure with a single axes
        daily_df = new_df.groupby('date').max().fillna(0)[['watering_event']].reset_index()
    
    # =============================================================================
    #     fig, ax = plt.subplots(figsize=(2, 2))
    #     chosen_month = daily_df.date.max().month
    #     print(chosen_month)
    # 
    #     ## Tell july to make a plot in a specific axes
    #     # july.month_plot(dates, data, month=2, date_label=True, ax=ax, colorbar=True)
    #     july.month_plot(daily_df.date, daily_df.watering_event,month=chosen_month, cmap="Blues",
    #                     date_label=True, ax=ax, fontsize=5,month_label=None, weeknum_label=False,   );
    #     plt.xticks(fontsize=5)
    # 
    #     ## Tell streamlit to display the figure
    #     st.pyplot(fig,use_container_width=True)
    # =============================================================================
        
    
        # This is going to be a switch to using the calplot library instead of july as that does not work when deployed to streamlit for hosting
        
        chosen_month = daily_df.date.max().month
        
        daily_df['month'] = pd.to_datetime(daily_df['date']).dt.month
        viz_daily_df = daily_df#[daily_df['month']==chosen_month]
        
        values = viz_daily_df.watering_event.values
        len(values)
        days = pd.to_datetime(viz_daily_df.date)
        len(days)
        events_tm = pd.Series(values, index=days)
        print(chosen_month)
    
        ## Tell july to make a plot in a specific axes
        # july.month_plot(dates, data, month=2, date_label=True, ax=ax, colorbar=True)
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_facecolor('black')
        #calplot.yearplot(events_tm, edgecolor='black', cmap='autumn', ax=ax, dropzero=True, fillcolor='whitesmoke', linecolor='black',linewidth=1.5)
        fig = calplot.calplot(events_tm, edgecolor='black', cmap='autumn',dropzero=True, fillcolor='whitesmoke', linecolor='black',linewidth=1.5,colorbar=False) 
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        ## Tell streamlit to display the figure
        st.pyplot(fig[0],use_container_width=True)        
        
        
        
        
        
        
        
        
        
        
        
        
    
        # the following sections preprocess the data for the model prediction
        
        df_2['date'] = df_2.created_at.str.split(' ').str[0]
        day_level_watering_df = df_2[['feed_id','date','watering_event']].groupby(['feed_id','date']).max().reset_index()
        
        list_of_feed_ids = [x for x in day_level_watering_df.feed_id.value_counts().index]
        new_df_container = []
        for n in range(len(list_of_feed_ids)):
            dataframe_to_consider = day_level_watering_df[day_level_watering_df.feed_id ==list_of_feed_ids[n]]
            for i in range(10):
                dataframe_to_consider['previous_date_'+str(-1*(i+1))] = dataframe_to_consider['date'].shift(i+1)
                dataframe_to_consider['previous_watering_event_value_'+str(-1*(i+1))] = dataframe_to_consider['watering_event'].shift(i+1)
                new_df_container.append(dataframe_to_consider)
          # day_level_watering_df['previous_date_'+str(-1*(i+1))] = pd.to_datetime(day_level_watering_df['previous_date_'+str(-1*(i+1))]).dt.day - 1
        day_level_watering_df = pd.concat(new_df_container)
        
        last_watered_dates = []
        for z in range(len(day_level_watering_df)):
            # calculate how many days it has been since last watered
            dates_array = (day_level_watering_df[day_level_watering_df.columns[1::2]].values[z])
        
            # print(dates_array)
        
            watering_events_flag_array = (day_level_watering_df[day_level_watering_df.columns[2::2]].values[z])
            # print(watering_events_flag_array)
        
            watering_event_indexes = [i for i in range(len(watering_events_flag_array)) if watering_events_flag_array[i] > 0]
            # print('the indexes which have been watered are', watering_event_indexes)
        
            # filter the list of dates based on the indexes above
            dates_last_watered = [dates_array[i] for i in watering_event_indexes]
            if len(dates_last_watered)<1:
                if z == 0:
                  # print('empty')
                  last_watered_dates.append(dates_array[0])
                elif z!=0: 
                    last_watered_dates.append(last_watered_dates[-1])
                continue
            elif len(dates_last_watered)>0 :
                date_last_watered = max(dates_last_watered)
                last_watered_dates.append(date_last_watered)
                
        day_level_watering_df['date_plant_last_watered'] = last_watered_dates
        days_since_last_watered_df = day_level_watering_df[~day_level_watering_df.date_plant_last_watered.isna()]
        days_since_last_watered_df['days_since_last_watered'] = pd.to_datetime(days_since_last_watered_df.date) - pd.to_datetime(days_since_last_watered_df.date_plant_last_watered)
        days_since_last_watered_df['days_since_last_watered'] = days_since_last_watered_df['days_since_last_watered'].astype(str).str.split(' ').str[0].astype(float)
        
        df_2 = df_2.merge(days_since_last_watered_df[['date','feed_id','days_since_last_watered']], on=['feed_id','date'],how='left')
    
    
        unique_feeds = [x for x in df_2.feed_id.value_counts().index]
        
        #st.dataframe(df)
        
        test_selected_plant_df = df_2[df_2.feed_id==selected_feed_id].sort_values('created_at').tail(1)
    
        # prepare the data to be input into the model
        
        columns_to_remove = ['id', 'soil_moisture_value', 'feed_id', 'created_at', 'name', 'key','previous_reading_time_value', 'time_delta_since_last_reading','previous_reading_value','date','soil_moisture_value_24hr_trailing',
       'change_vs_soil_moisture_value_24hr_trailing',
       'soil_moisture_value_48hr_trailing',
       'change_vs_soil_moisture_value_48hr_trailing']
        columns_to_keep = [x for x in df_2.columns if x not in columns_to_remove]
    
        valid_df = test_selected_plant_df[columns_to_keep].copy()
        
        independent_variables = [x for x in valid_df.columns if x!='watering_event']
        dependent_variable = 'watering_event'
    
        input_df = valid_df[columns_to_keep].copy().fillna(0)
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df = input_df.dropna()
        #st.dataframe(input_df[independent_variables])
        input_X_values = input_df[independent_variables].drop(columns='plant_name').values
        
        
        # Load pre-trained classification model
        model = load_model('water_today_model.h5')
        prediction = model.predict(input_X_values)
        prediction_rounded = prediction.round(0)
        
        if prediction_rounded==1:
            model_prediction_text = f'We think you should check this plant to see if it needs to be watered.'
        else:
            model_prediction_text = f"Your plant seems to be doing great and doesn't need to be watered right now."


































    with watering_date_text_box:
        st.write(date_last_watered_sentence)
        if len(watering_df)>1:
            if average_number_of_days_between_waterings>=2:
                if average_number_of_days_between_waterings<30:
                    st.write('Your ' + selected_plant.replace('_',' ').title() + ' is typically being watered every ' +str(int(average_number_of_days_between_waterings))+' days.')
                    if prediction_rounded==1:
                        st.markdown(f':orange[{model_prediction_text}]')
                    elif prediction_rounded==0:
                        st.markdown(f':blue[{model_prediction_text}]')

        else:
            st.write(' ')
        
        
# the code below generates a line graph showing the daily min max and
# mean values for the moisture readings of the selected plant
# =============================================================================
#     viz_df = df.melt(id_vars='date',var_name='moisture_reading_type')
#     st.header('Moisture readings over time')
#     fig = px.line(viz_df[(viz_df.moisture_reading_type.str.contains('mean_value'))|
#                          (viz_df.moisture_reading_type.str.contains('min_value'))], x='date',y='value',color='moisture_reading_type')
#     if len(watering_df)>0:
#         for i in range(len(watering_df)):
#             fig.add_vline(x=watering_df.reset_index().date[i], line_width=3, line_dash="dash", line_color="green")
#     st.plotly_chart(fig, theme='streamlit', use_container_width=True)
# =============================================================================
# this new chart is based on the soil moisture absorption rate which seems to be a much simpler approach to understanding watering events

    viz_df = new_df.melt(id_vars='date',var_name='moisture_reading_type')
    viz_df = viz_df[viz_df['moisture_reading_type']=='soil_moisture_absorption_rate']
    viz_df = viz_df[['date','value']].groupby(['date']).mean().reset_index()
    viz_df.date = viz_df.date.astype(str)
    viz_df.value = viz_df.value.fillna(np.nan).astype(float)
    # create a full date rage to account for days when meter readings fail
    
    full_date_ranges = pd.date_range(start=viz_df.date.min(),end=viz_df.date.max())
    full_date_range_df = pd.DataFrame(full_date_ranges)
    full_date_range_df.columns = ['date']
    full_date_range_df['date'] = full_date_range_df.date.astype(str).str.split(' ').str[0]
    viz_df = full_date_range_df.merge(viz_df, on='date', how='left')
    viz_df.date = pd.to_datetime(viz_df.date)
    viz_df = viz_df.set_index('date')
    viz_df.value = viz_df.value.fillna(np.nan).astype(float)

    viz_df['value'] = viz_df['value'].astype(float).interpolate(method='pad')
    viz_df['value'] = viz_df['value'].astype(float)
    viz_df['value'] = viz_df['value'].astype(str).fillna('0').str.replace('inf','0').str.replace('-inf','0').astype(float)
    viz_df = viz_df.reset_index()
    #st.markdown('pandas version=='+pd.__version__)
    st.header('Moisture absorption rate')
    fig = px.line(viz_df, x='date',y='value')
# =============================================================================
#     if len(watering_df)>0:
#         for i in range(len(watering_df)):
#             fig.add_vline(x=watering_df.reset_index().date[i], line_width=3, line_dash="dash", line_color="green")
# =============================================================================
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
        ))
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    
    
    st.header('Soil moisture levels')
    
    viz_df = new_df.melt(id_vars='date',var_name='moisture_reading_type')
    viz_df = viz_df[viz_df['moisture_reading_type']=='soil_moisture_value']
    viz_df = viz_df[['date','value']].groupby(['date']).mean().reset_index()
    viz_df.date = viz_df.date.astype(str)
    
    # create a full date rage to account for days when meter readings fail
    
    full_date_ranges = pd.date_range(start=viz_df.date.min(),end=viz_df.date.max())
    full_date_range_df = pd.DataFrame(full_date_ranges)
    full_date_range_df.columns = ['date']
    full_date_range_df['date'] = full_date_range_df.date.astype(str).str.split(' ').str[0]
    viz_df = full_date_range_df.merge(viz_df, on='date', how='left')
    viz_df.date = pd.to_datetime(viz_df.date)
    viz_df = viz_df.set_index('date')
    viz_df.value = viz_df.value.fillna(np.nan).astype(float)

    viz_df['value'] = viz_df['value'].astype(float).interpolate(method='pad')
    viz_df['value'] = viz_df['value'].astype(float)
    viz_df['value'] = viz_df['value'].astype(str).fillna('0').str.replace('inf','0').str.replace('-inf','0').astype(float)
    viz_df = viz_df.reset_index()
    fig = px.line(viz_df, x='date',y='value')
# =============================================================================
#     if len(watering_df)>0:
#         for i in range(len(watering_df)):
#             fig.add_vline(x=watering_df.reset_index().date[i], line_width=3, line_dash="dash", line_color="green")
# =============================================================================
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
        ))
    fig.update_layout(yaxis_range=[0, 110])

    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

