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
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


st.header("Crypto Price Prediction")
st.info(
    "This Will Give Analysis on the present and last ten year prices of crypto coins"
)

# Input for coin name
coin_name = st.text_input("Enter the cryptocurrency symbol (e.g., BTC, ETH)", "BTC")


def remove(x):
    """
    This function will strip the data column of the dataframe.
    """
    x = str(x)
    res = x.split(" ")[0]
    return res


data = yf.download(coin_name + "-USD", period="max")

st.subheader("DataSet")
st.write(data)

data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
data.index = data.index.to_series().apply(
    lambda x: remove(x)
)  # applying preprocessing function

Eda, DataPreprocessing, Models = st.tabs(
    ["Exploratory Data Analysis", "Data PreProcessing", "Models"]
)

with Eda:
    st.subheader("Shape")
    shape = data.shape
    st.write(shape)

    st.subheader("Head")
    head = data.head()
    st.write(head)

    st.subheader("Tail")
    tail = data.tail()
    st.write(tail)

    st.subheader("Info")
    info = data.info
    st.write(info)

    st.subheader("Describe")
    describe = data.describe()
    st.write(describe)

    st.subheader("Index")
    index = data.index
    st.write(index)

    st.subheader("Null Values")
    st.write("Null Values:", data.isnull().values.sum())
    st.write("NA values:", data.isnull().values.any())
    st.info("This Shows that there are no null values")

    st.title("Year Wise Distribution Of The DataSet")

    new_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Define functions specific to EDA tab
    def yearly_analysis(start_date, end_date):
        year = data.loc[start_date:end_date]
        year.index = pd.to_datetime(year.index, format="%Y-%m-%d")
        return year

    def yearly_chart(yearlyDataset):
        names = cycle([coin_name + " Close Price", coin_name])

        fig = px.line(
            yearlyDataset,
            x=yearlyDataset.index,
            y="Close",
            labels={"Date": "Date", "value": coin_name + " value"},
        )
        fig.update_layout(
            title_text=coin_name + " analysis chart",
            font_size=15,
            font_color="black",
            legend_title_text=coin_name + " Parameters",
        )
        fig.for_each_trace(lambda t: t.update(name=next(names)))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig)

        figure = go.Figure(
            data=[
                go.Candlestick(
                    x=yearlyDataset.index,
                    low=yearlyDataset["Low"],
                    high=yearlyDataset["High"],
                    close=yearlyDataset["Close"],
                    open=yearlyDataset["Open"],
                    increasing_line_color="green",
                    decreasing_line_color="red",
                )
            ]
        )
        st.plotly_chart(figure)

    def every_year_monthwise_analysis(yearlyDataset):
        monthwise = yearlyDataset.groupby(yearlyDataset.index.strftime("%B"))[
            ["Open", "Close"]
        ].mean()
        monthwise = monthwise.reindex(new_order, axis=0)
        return monthwise

    def monthly_open_close_chart(monthlyDataset):
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=monthlyDataset.index,
                y=monthlyDataset["Open"],
                name=coin_name + " Open Price",
                marker_color="crimson",
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthlyDataset.index,
                y=monthlyDataset["Close"],
                name=coin_name + " Close Price",
                marker_color="lightsalmon",
            )
        )

        fig.update_layout(
            barmode="group",
            xaxis_tickangle=-45,
            title="Monthwise comparison between " + coin_name + " open and close price",
        )
        st.plotly_chart(fig)

    def monthly_high_low_chart(yearlyDataset):
        yearlyDataset.groupby(yearlyDataset.index.strftime("%B"))["Low"].min()
        monthwise_high = yearlyDataset.groupby(yearlyDataset.index.strftime("%B"))[
            "High"
        ].max()
        monthwise_high = monthwise_high.reindex(new_order, axis=0)

        monthwise_low = yearlyDataset.groupby(yearlyDataset.index.strftime("%B"))[
            "Low"
        ].min()
        monthwise_low = monthwise_low.reindex(new_order, axis=0)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthwise_high.index,
                y=monthwise_high,
                name=coin_name + " High Price",
                marker_color="rgb(0, 153, 204)",
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthwise_low.index,
                y=monthwise_low,
                name=coin_name + " Low Price",
                marker_color="rgb(255, 128, 0)",
            )
        )

        fig.update_layout(
            barmode="group", title="Monthwise High and Low " + coin_name + " price"
        )
        st.plotly_chart(fig)

    # Input for start and end dates
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")

    if start_date and end_date:
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

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
        data_copy.index = pd.to_datetime(data_copy.index, format="%Y-%m-%d")
        data_copy["Month"] = data_copy.index.month
        data_copy["Year"] = data_copy.index.year
        grouped_data = (
            data_copy.groupby(["Year", "Month"])["Volume"].sum().reset_index()
        )
        grouped_data["Month"] = grouped_data["Month"].apply(
            lambda x: calendar.month_name[x]
        )
        x = grouped_data.pivot_table(
            index="Year", columns="Month", values="Volume", aggfunc="sum"
        )
        x.columns = pd.Categorical(x.columns, categories=new_order, ordered=True)
        x = x.sort_index(axis=1)

        # Plotting heatmap with Plotly
        fig_heatmap = go.Figure(
            data=go.Heatmap(z=x.values, x=x.columns, y=x.index, colorscale="Viridis")
        )
        fig_heatmap.update_layout(
            title=f"Heatmap of Monthly Volume for {coin_name}",
            xaxis_title="Month",
            yaxis_title="Year",
        )
        st.plotly_chart(fig_heatmap)

        DSR = data["Close"].pct_change(1)
        st.write(DSR)
        st.write(DSR.describe())
        st.info(
            f"if we invest in {coin_name}, we can approx get 0.2% return daily  investment has gome to 37% daily loss sometimes investment has gone to 25% daily profit sometimes ( max price increased in a day is 25% "
        )


with DataPreprocessing:
    st.title("Feature Engineering")

    def SMA(dataset, period=30, column="Close"):
        return dataset[column].rolling(window=period).mean()

    def EMA(dataset, period=20, column="Close"):
        return dataset[column].ewm(span=period, adjust=False).mean()

    def MACD(data, period_long=26, period_short=12, period_signal=9, column="Close"):
        ShortEMA = EMA(data, period_short, column=column)
        LongEMA = EMA(data, period_long, column=column)
        data["MACD"] = ShortEMA - LongEMA
        data["Signal_Line"] = EMA(data, period_signal, column="MACD")
        return data

    def RSI(data, period=14, column="Close"):
        delta = data[column].diff(1)
        delta = delta[1:]
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        data["up"] = up
        data["down"] = down
        AVG_Gain = SMA(data, period, column="up")
        AVG_Loss = abs((SMA(data, period, column="down")))
        RS = AVG_Gain / AVG_Loss
        RSI = 100.0 - (100.0 / (1.0 + RS))
        data["RSI"] = RSI
        return data

    MACD(data)
    RSI(data)
    data["SMA"] = SMA(data)
    data["EMA"] = EMA(data)

    st.write(data)

    st.title("Scaling The Data")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    st.write(scaled_df)


with Models:

    WPP, PP = st.tabs(["Without Python Package", "With Python Package"])

    with PP:
        future_days = st.text_input("Enter the days for future prediction")
        if future_days:
            
            try:
                # Convert input to integer
                future_days = int(future_days)

                # Create a new column for future predictions
                data[str(future_days) + "_Days_Price_Pred"] = data[["Close"]].shift(
                    -future_days
                )

                # Select relevant columns to display
                future_data = data[["Close", str(future_days) + "_Days_Price_Pred"]]

                # Display the DataFrame
                st.write(future_data)
            except ValueError:
                st.error("Invalid input. Please enter a valid number of days.")

            st.header("Normal Data Using only Close Column")
            columns_to_extract = ["Close"]  # specify the columns you want to extract
            X = np.array(data[columns_to_extract])
            X = X[: data.shape[0] - future_days]
            y = np.array(data[str(future_days) + "_Days_Price_Pred"])
            y = y[:-future_days]
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

            st.subheader("SVM MODEL")
            svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.00001)
            svr_rbf.fit(X_train, y_train)
            svr_rbf_confidence = svr_rbf.score(X_test, y_test)
            st.write(svr_rbf_confidence)

            st.subheader("Linear Regression")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_confidence = lr_model.score(X_test, y_test)
            st.write(lr_confidence)

            st.subheader("Random Forest")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_confidence = rf_model.score(X_test, y_test)
            st.write(rf_confidence)

            st.header("Normal Data Using All Columns")
            columns_to_extract = [
                "Close",
                "MACD",
                "RSI",
                "SMA",
                "EMA",
            ]  # specify the columns you want to extract
            X = np.array(data[columns_to_extract])
            X = X[30 : data.shape[0] - future_days]
            y = np.array(data[str(future_days) + "_Days_Price_Pred"])
            y = y[30:-future_days]
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

            st.subheader("Svm model")
            svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.00001)
            svr_rbf.fit(X_train, y_train)
            svr_rbf_confidence = svr_rbf.score(X_test, y_test)
            st.write(svr_rbf_confidence)

            st.subheader("Linear Regression  model")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_confidence = lr_model.score(X_test, y_test)
            st.write(lr_confidence)

            st.subheader("Random Forest model")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_confidence = rf_model.score(X_test, y_test)
            st.write(rf_confidence)

            st.header("Scaled Data Using Only Close Column")
            try:
                future_days = int(future_days)  # Convert to integer

                # Create a new column for future predictions
                scaled_df[str(future_days) + "_Days_Price_Pred"] = scaled_df[
                    "Close"
                ].shift(-future_days)

                # Drop rows with NaN values
                scaled_df = scaled_df.dropna(
                    subset=[str(future_days) + "_Days_Price_Pred"]
                )

                # Select relevant columns to display
                future_data = scaled_df[
                    ["Close", str(future_days) + "_Days_Price_Pred"]
                ]

            except ValueError:
                st.error("Invalid input. Please enter a valid number of days.")

            columns_to_extract = ["Close"]  # specify the columns you want to extract
            X = np.array(scaled_df[columns_to_extract])
            X = X[: scaled_df.shape[0] - future_days]
            y = np.array(scaled_df[str(future_days) + "_Days_Price_Pred"])
            y = y[:-future_days]
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

            st.subheader("Svm Model")
            svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.00001)
            svr_rbf.fit(X_train, y_train)
            svr_rbf_confidence = svr_rbf.score(X_test, y_test)
            st.write(svr_rbf_confidence)

            st.subheader("Linear Regression")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_confidence = lr_model.score(X_test, y_test)
            st.write(lr_confidence)

            st.subheader("Random Forest")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_confidence = rf_model.score(X_test, y_test)
            st.write(rf_confidence)

            st.header("Scaled Data - Using All Columns")
            columns_to_extract = [
                "Close",
                "MACD",
                "RSI",
                "SMA",
                "EMA",
            ]  # specify the columns you want to extract
            X = np.array(scaled_df[columns_to_extract])
            X = X[30 : scaled_df.shape[0] - future_days]
            y = np.array(scaled_df[str(future_days) + "_Days_Price_Pred"])
            y = y[30:-future_days]
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

            st.subheader("Svm Model")
            svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.00001)
            svr_rbf.fit(X_train, y_train)
            svr_rbf_confidence = svr_rbf.score(X_test, y_test)
            st.write(svr_rbf_confidence)

            st.subheader("Linear Regression")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_confidence = lr_model.score(X_test, y_test)
            st.write(lr_confidence)

            st.subheader("Random Forest")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_confidence = rf_model.score(X_test, y_test)
            st.write(rf_confidence)

            st.header("Evaluation Metrics")
            st.subheader("For All Models")
            svm_predictions = svr_rbf.predict(X_test)
            lr_predictions = lr_model.predict(X_test)
            rf_predictions = rf_model.predict(X_test)

            def evaluate_model(y_test, predictions):
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                return mse, mae, r2, mape

            lr_mse, lr_mae, lr_r2, lr_mape = evaluate_model(y_test, lr_predictions)
            rf_mse, rf_mae, rf_r2, rf_mape = evaluate_model(y_test, rf_predictions)
            svm_mse, svm_mae, svm_r2, svm_mape = evaluate_model(y_test, svm_predictions)

            st.write(
                f"""Linear Regression -
            MSE: {lr_mse},
            MAE: {lr_mae},
            R2: {lr_r2},
            MAPE: {lr_mape}%

            """
            )

            st.write(
                f"""Random Forest -
            MSE: {rf_mse},
            MAE: {rf_mae},
            R2: {rf_r2},
            MAPE: {rf_mape}%

            """
            )

            st.write(
                f"""SVM -
            MSE: {svm_mse},
            MAE: {svm_mae},
            R2: {svm_r2},
            MAPE: {svm_mape}%

            """
            )

            st.header("Plotting Predictions Vs Actual")
            # Plotting the results
            plt.figure(figsize=(14, 7))

            # Linear Regression
            plt.subplot(3, 1, 1)
            plt.plot(y_test, label="Actual", color="blue")
            plt.plot(lr_predictions, label="Linear Regression", color="orange")
            plt.title("Linear Regression vs Actual")
            plt.legend()

            # Random Forest
            plt.subplot(3, 1, 2)
            plt.plot(y_test, label="Actual", color="blue")
            plt.plot(rf_predictions, label="Random Forest", color="green")
            plt.title("Random Forest vs Actual")
            plt.legend()

            # SVM
            plt.subplot(3, 1, 3)
            plt.plot(y_test, label="Actual", color="blue")
            plt.plot(svm_predictions, label="SVM", color="red")
            plt.title("SVM vs Actual")
            plt.legend()

            plt.tight_layout()
            st.pyplot(plt)
