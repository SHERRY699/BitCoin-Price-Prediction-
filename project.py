import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
import datetime as dt
import streamlit as st 
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import calendar

st.header("Crypto Price Prediction")
st.info("This Will Give Analysis on the present and last ten year prices of crypto coins")

# Input for coin name
coin_name = st.text_input("Enter the cryptocurrency symbol (e.g., BTC, ETH)", "BTC")

def remove(x):
    """
    This function will strip the data column of the dataframe.
    """
    x = str(x)
    res = x.split(" ")[0]
    return res

data = yf.download(coin_name+"-USD", period="max")

st.subheader('DataSet')
st.write(data)

data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
data.index = data.index.to_series().apply(lambda x: remove(x))  # applying preprocessing function

Eda, DataPreprocessing, Models = st.tabs(['Exploratory Data Analysis','Data PreProcessing','Models'])

with Eda:
    st.subheader('Shape')
    shape = data.shape
    st.write(shape)

    st.subheader('Head')
    head = data.head()
    st.write(head)

    st.subheader('Tail')
    tail = data.tail()
    st.write(tail)

    st.subheader('Info')
    info = data.info
    st.write(info)

    st.subheader('Describe')
    describe = data.describe()
    st.write(describe)

    st.subheader('Index')
    index = data.index
    st.write(index)

    st.subheader('Null Values')
    st.write('Null Values:', data.isnull().values.sum())
    st.write('NA values:', data.isnull().values.any())
    st.info("This Shows that there are no null values")

    st.title('Year Wise Distribution Of The DataSet')

    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                 'September', 'October', 'November', 'December']

    # Define functions specific to EDA tab
    def yearly_analysis(start_date, end_date):
        year = data.loc[start_date:end_date]
        year.index = pd.to_datetime(year.index, format='%Y-%m-%d')
        return year

    def yearly_chart(yearlyDataset):
        names = cycle([coin_name + ' Close Price', coin_name])
        
        fig = px.line(yearlyDataset, x=yearlyDataset.index, y='Close',
                    labels={'Date': 'Date', 'value': coin_name + ' value'})
        fig.update_layout(title_text=coin_name + ' analysis chart', font_size=15, font_color='black', legend_title_text=coin_name + ' Parameters')
        fig.for_each_trace(lambda t: t.update(name=next(names)))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig)
        
        figure = go.Figure(
            data=[
                go.Candlestick(
                    x=yearlyDataset.index,
                    low=yearlyDataset['Low'],
                    high=yearlyDataset['High'],
                    close=yearlyDataset['Close'],
                    open=yearlyDataset['Open'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )
            ]
        )
        st.plotly_chart(figure)

    def every_year_monthwise_analysis(yearlyDataset):
        monthwise = yearlyDataset.groupby(yearlyDataset.index.strftime('%B'))[['Open', 'Close']].mean()
        monthwise = monthwise.reindex(new_order, axis=0)
        return monthwise

    def monthly_open_close_chart(monthlyDataset):
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthlyDataset.index,
            y=monthlyDataset['Open'],
            name=coin_name + ' Open Price',
            marker_color='crimson'
        ))
        fig.add_trace(go.Bar(
            x=monthlyDataset.index,
            y=monthlyDataset['Close'],
            name=coin_name + ' Close Price',
            marker_color='lightsalmon'
        ))
        
        fig.update_layout(barmode='group', xaxis_tickangle=-45, 
                        title='Monthwise comparison between ' + coin_name + ' open and close price')
        st.plotly_chart(fig)

    def monthly_high_low_chart(yearlyDataset):
        yearlyDataset.groupby(yearlyDataset.index.strftime('%B'))['Low'].min()
        monthwise_high = yearlyDataset.groupby(yearlyDataset.index.strftime('%B'))['High'].max()
        monthwise_high = monthwise_high.reindex(new_order, axis=0)
        
        monthwise_low = yearlyDataset.groupby(yearlyDataset.index.strftime('%B'))['Low'].min()
        monthwise_low = monthwise_low.reindex(new_order, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthwise_high.index,
            y=monthwise_high,
            name=coin_name + ' High Price',
            marker_color='rgb(0, 153, 204)'
        ))
        fig.add_trace(go.Bar(
            x=monthwise_low.index,
            y=monthwise_low,
            name=coin_name + ' Low Price',
            marker_color='rgb(255, 128, 0)'
        ))
        
        fig.update_layout(barmode='group', title='Monthwise High and Low ' + coin_name + ' price')
        st.plotly_chart(fig)

    # Input for start and end dates
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")

    if start_date and end_date:
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    
    if st.button("Analyze"):
        year_2014 = yearly_analysis(start_date, end_date)
        monthly_2014 = every_year_monthwise_analysis(year_2014)
        
        st.subheader(f"Yearly Chart for {coin_name}")
        yearly_chart(year_2014)
        
        st.subheader(f"Monthly Open and Close Chart for {coin_name}")
        monthly_open_close_chart(monthly_2014)
        
        st.subheader(f"Monthly High and Low Chart for {coin_name}")
        monthly_high_low_chart(year_2014)
        
        st.subheader(f"Heatmap of Monthly Volume for {coin_name}")
        data_copy = data.copy()
        data_copy.index = pd.to_datetime(data_copy.index, format='%Y-%m-%d')
        data_copy['Month'] = data_copy.index.month
        data_copy['Year'] = data_copy.index.year
        grouped_data = data_copy.groupby(['Year', 'Month'])['Volume'].sum().reset_index()
        grouped_data['Month'] = grouped_data['Month'].apply(lambda x: calendar.month_name[x])
        x = grouped_data.pivot_table(index='Year', columns='Month', values='Volume', aggfunc='sum')
        x.columns = pd.Categorical(x.columns, categories=new_order, ordered=True)
        x = x.sort_index(axis=1)

    # Plotting heatmap with Plotly
        fig_heatmap = go.Figure(data=go.Heatmap(
        z=x.values,
        x=x.columns,
        y=x.index,
        colorscale='Viridis'
    ))
        fig_heatmap.update_layout(
        title=f"Heatmap of Monthly Volume for {coin_name}",
        xaxis_title='Month',
        yaxis_title='Year'
    )
        st.plotly_chart(fig_heatmap)

with DataPreprocessing:
    st.write("Data Preprocessing tab content goes here")

with Models:
    st.write("Models tab content goes here")
