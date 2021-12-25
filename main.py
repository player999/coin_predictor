import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import argparse, sys
from datetime import datetime
import queue

def load_data(fname: str)->np.array:
    data_schema = {
        "Open time": 0,
        "Open": 1,
        "High": 2,
        "Low": 3,
        "Close": 4,
        "Volume": 5,
        "Close time": 6,
        "Quote asset volume": 7,
        "Number of trades": 8,
        "Taker buy base asset volume": 9,
        "Taker buy quote asset volume": 10,
        "Ignore": 11
    }

    df = pd.read_csv(filename, header=None)

    def convert_to_date(x, **kwargs):
        t = datetime.fromtimestamp(x / 1000)
        return t

    df[data_schema["Open time"]] = df[data_schema["Open time"]].apply(convert_to_date)
    df.set_axis(df[data_schema["Open time"]], inplace=True)

    data_real = np.dstack((df[data_schema["High"]][:-1].values,
                           df[data_schema["Open"]][:-1].values,
                           df[data_schema["Close"]][:-1].values,
                           df[data_schema["Low"]][:-1].values))[0]

    high_data = np.divide(df[data_schema["High"]][1:].values, df[data_schema["High"]][:-1].values)
    open_data = np.divide(df[data_schema["Open"]][1:].values, df[data_schema["Open"]][:-1].values)
    close_data = np.divide(df[data_schema["Close"]][1:].values, df[data_schema["Close"]][:-1].values)
    low_data = np.divide(df[data_schema["Low"]][1:].values, df[data_schema["Low"]][:-1].values)

    data = np.dstack((high_data, open_data, close_data, low_data))[0]
    return (df[data_schema["Open time"]], data, data_real)

def apply_prediction_to_data(prediction, input_data, look_back):
    return np.multiply(input_data[look_back:], prediction)

def train_model(train_data, look_back, model_fname):
    train_generator = TimeseriesGenerator(train_data[1], train_data[1], length=look_back, batch_size=10000)
    model = Sequential()
    model.add(
        LSTM(10,
             activation='sigmoid',
             input_shape=(look_back, 4))
    )
    model.add(Dense(50))
    model.add(Dense(4))
    model.compile(optimizer='sgd', loss='mse')

    num_epochs = 100
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
    model.save(model_fname)
    return model

def train_model_close_rise(train_data, look_back, model_fname):
    tdata = np.array(list(map(lambda x: x[2], train_data[1])))
    tdata.reshape((-1, 1))
    train_generator = TimeseriesGenerator(tdata, tdata, length=look_back, batch_size=10000)
    model = Sequential()
    model.add(
        LSTM(10,
             activation='relu',
             input_shape=(look_back, 1))
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 100
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
    model.save(model_fname)
    return model

def do_trade(trading_state, input_data):
    if len(input_data) != (trading_state["lookback"] + 1):
        return trading_state
    input_data = np.array(list(map(lambda x: x[2], input_data)))
    model_input = np.array([np.divide(input_data[1:], input_data[:-1])])
    prediction = trading_state["model"].predict(model_input)[0]
    prediction_close = prediction[0]
    input_close = input_data[-1]

    if trading_state["crypto_currency"] > 0:
        trading_state["normal_currency"] = input_close * trading_state["crypto_currency"]
        trading_state["crypto_currency"] = 0
    if prediction_close > 1.0:
        trading_state["crypto_currency"] = trading_state["normal_currency"] / input_close
        trading_state["normal_currency"] = 0
    return trading_state

def do_trade4(trading_state, input_data):
    if len(input_data) != (trading_state["lookback"] + 1):
        return trading_state
    input_data = np.array(input_data)
    model_input = np.array([np.divide(input_data[1:], input_data[:-1])])
    prediction = trading_state["model"].predict(model_input)[0]
    prediction_close = prediction[2]
    input_close = input_data[-1][2]

    if trading_state["crypto_currency"] > 0:
        trading_state["normal_currency"] = input_close * trading_state["crypto_currency"]
        trading_state["crypto_currency"] = 0
    if prediction_close > 1.0:
        trading_state["crypto_currency"] = trading_state["normal_currency"] / input_close
        trading_state["normal_currency"] = 0
    return trading_state

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or apply prediction model')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--predict', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--trade-benchmark', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--model-file', type=str, default="data/ethusdtmodel.mod")
    parser.add_argument('--train-data', type=str, default="data/total.csv")
    parser.add_argument('--benchmark-input', type=str, default="data/ETHUSDT-15m-2021-12-22.csv")

    args = parser.parse_args()
    filename = args.train_data

    (times, data, data_real) = load_data(filename)
    split_percent = 0.995
    split = int(split_percent * len(data))

    data_train = data[:split]
    data_test = data[split:]
    data_real_test = data_real[split:]

    date_train = times[:split]
    date_test = times[split:]

    look_back = 5

    if args.train:
        model = train_model((date_train, data_train), look_back, args.model_fname)

    if args.predict:
        model = load_model(args.model_fname)
        test_generator = TimeseriesGenerator(data_test, data_test, length=look_back, batch_size=1)
        prediction = model.predict_generator(test_generator)
        prediction_decoded = apply_prediction_to_data(prediction, data_real_test, look_back)
        points_count = 500

        prediction_time = date_test[look_back+1:]
        trace2 = go.Candlestick(
            x=prediction_time,
            high=list(map(lambda x: x[0], prediction_decoded)),
            open=list(map(lambda x: x[1], prediction_decoded)),
            close=list(map(lambda x: x[2], prediction_decoded)),
            low=list(map(lambda x: x[3], prediction_decoded)),
            increasing_line_color='cyan', decreasing_line_color='gray',
            name='Prediction'
        )
        trace3 = go.Candlestick(
            x=date_test,
            high=list(map(lambda x: x[0], data_real_test)),
            open=list(map(lambda x: x[1], data_real_test)),
            close=list(map(lambda x: x[2], data_real_test)),
            low=list(map(lambda x: x[3], data_real_test)),
            name='Ground Truth'
        )

        layout = go.Layout(
            title="ETHUSDT",
            xaxis={'title': "Open time"},
        )

        layout_raw = go.Layout(
            title="ETHUSDT",
            xaxis={'title': "Open time"},
        )

        prediction_dat_high = go.Scatter(x=date_test, y=list(map(lambda x: x[0], prediction)))
        prediction_dat_open = go.Scatter(x=date_test, y=list(map(lambda x: x[1], prediction)))
        prediction_dat_close = go.Scatter(x=date_test, y=list(map(lambda x: x[2], prediction)))
        prediction_dat_low = go.Scatter(x=date_test, y=list(map(lambda x: x[3], prediction)))
        fig1 = go.Figure(data=[prediction_dat_high, prediction_dat_open, prediction_dat_close, prediction_dat_low], layout=layout_raw)
        fig2 = go.Figure(data=[trace2, trace3], layout=layout)
        fig1.show()
        fig2.show()

    if args.trade_benchmark:
        model = load_model(args.model_file)
        filename = args.benchmark_input
        (times, _, data_real) = load_data(filename)
        input_queue = queue.Queue(maxsize=look_back + 1)
        trading_state = {}
        trading_state["lookback"] = look_back
        trading_state["model"] = model
        trading_state["normal_currency"] = 50
        trading_state["crypto_currency"] = 0
        for el in data_real:
            input_queue.put(el)
            dat = list(input_queue.queue)
            dat.reverse()
            dat_np = np.array(dat)
            trading_state = do_trade4(trading_state, dat_np)
            if input_queue.full():
                input_queue.get()

        if trading_state["crypto_currency"] > 0:
            trading_state["normal_currency"] = data_real[-1][2] * trading_state["crypto_currency"]
            trading_state["crypto_currency"] = 0

        print("Normal currency: %f" % (trading_state["normal_currency"]))
        print("Crypto currency: %f" % (trading_state["crypto_currency"]))

