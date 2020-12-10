
import datetime as dt
import numpy as np
import pandas as pd
import cudf
import matplotlib.pyplot as plt
import logging
import time
import os.path
import sys
import fill_trading_data
import DB_config

from DB_config import make_engine
from pandas_datareader import data as pdr 
from pandas import DataFrame
from datetime import timedelta
from datetime import datetime
from mysql.connector import Error
from sklearn import preprocessing



def get_DSU_table():
	engine = make_engine("StockInfo")
	columns = []
	dsuDF = pd.DataFrame()

	#for some reason python3 doesn't like me calling stored procedures, so lets just go with this,
	# we only need to do its once anyways

	get_distinct_stock_updates1 = "select max(id) id, Ticker, start_date, end_date, daily_avg_price, daily_avg_volume, avg_daily_high, "
	get_distinct_stock_updates2 = "avg_daily_low, avg_daily_volatility_percentage, num_data_points, avg_30m_volatility_percentage "
	get_distinct_stock_updates3 = "from DailyStockUpdate "
	get_distinct_stock_updates4 = "group by Ticker, start_date, end_date, daily_avg_price, daily_avg_volume, avg_daily_high, "
	get_distinct_stock_updates5 = "avg_daily_low, avg_daily_volatility_percentage, num_data_points, avg_30m_volatility_percentage " 
	get_distinct_stock_updates6 = "order by Ticker"

	get_distinct_stock_updates_QUERY = get_distinct_stock_updates1+get_distinct_stock_updates2+get_distinct_stock_updates3+get_distinct_stock_updates4+get_distinct_stock_updates5+get_distinct_stock_updates6


	with engine.connect() as connection:
		columns = connection.execute("SHOW columns FROM {}".format("DailyStockUpdate"))
		columns = [r[0] for r in columns]

	with engine.connect() as connection:
		dsuDF = pd.DataFrame(connection.execute(get_distinct_stock_updates_QUERY), columns=columns)

	dsuDF = dsuDF.drop('id', axis=1)
	print(len(dsuDF))
	engine.dispose() 
	return(dsuDF)


def get_tickers_data(ticker):
	engine = make_engine("StockInfo")

	with engine.connect() as connection:
		columns = connection.execute("SHOW columns FROM {}".format(ticker))
		columns = [r[0] for r in columns]

		tickerDf = pd.DataFrame(connection.execute("SELECT * FROM StockInfo.{}".format(ticker)), columns=columns)
		tickerDf = tickerDf.drop('id', axis=1)
		
		tickerDf.datetime = tickerDf.datetime.astype(str)
		
		#converts datetime column to to datetime.datetime type
		for row in range(len(tickerDf)):
			tickerDf.loc[row, 'datetime'] = dt.datetime.strptime(tickerDf.loc[row, 'datetime'] , '%Y-%m-%d %H:%M:%S')

		tickerDf['date'] = [datetime.date() for datetime in tickerDf['datetime']]
		tickerDf['time'] = [datetime.time() for datetime in tickerDf['datetime']]
		tickerDf = tickerDf.drop('datetime', axis=1)
		dfCols = tickerDf.columns.tolist()
		dfCols = dfCols[-2:] + dfCols[:-2]
		tickerDf = tickerDf[dfCols]

	engine.dispose()
	
	return tickerDf

		#apparently the cudf is slower than the pandas for this operation, so no point in using it

		#cudf_df = cudf.DataFrame.from_pandas(aaDf) #datetime.datetime values not supported

		#dt_obj= '2020-09-18 09:30:00'
		#dt_obj = dt.datetime.strptime(dt_obj, '%Y-%m-%d %H:%M:%S')
		#print(dt_obj)
		#print(type(dt_obj))
		#exists = dt_obj in tickerDf.datetime.values
		#exists = dt_obj in cudf_df.datetime.values

		#print(exists)


def get_avg_30m_volatility(days_data):
	currentTime_uncut = dt.datetime.strptime('9:30:00', '%H:%M:%S')
	intervalEndTime_uncut = currentTime_uncut + timedelta(minutes=30)
	days_last_recorded_time = days_data['time'].iloc[-1]
	days_last_close = days_data['close'].iloc[-1]

	sum30mSTD = 0
	interval_volatility_percentage = 0

	num_periods = 0

	#Calculate avg_30m volatility
	# there is 13 max 30m periods
	for currTime in range(13):
		currentTime = currentTime_uncut.time()
		intervalEndTime = intervalEndTime_uncut.time()
		num_periods += 1

		intervalData = days_data.loc[(days_data.time >= currentTime) & (days_data.time < intervalEndTime)]
		#print(intervalData)
		interval_30m_std = intervalData['close'].std() 
		interval_last_close = intervalData['close'].iloc[-1]
		
		interval_volatility_percentage +=(interval_30m_std*2)/interval_last_close
		
		if(days_last_recorded_time <= intervalEndTime):
			break
		
		
		currentTime_uncut += timedelta(minutes=30)
		intervalEndTime_uncut += timedelta(minutes=30)
	
	avg_30m_volatility_percentage_1 = interval_volatility_percentage/num_periods
	#print(avg_30m_volatility_percentage_1)
	# input("calc at end of interval: " + str(avg_30m_volatility_percentage_1))

	# avg_30m_volatility = ((sum30mSTD/num_periods)*2)/days_last_close
	# input("calc at end of day: " + str(avg_30m_volatility))

	return avg_30m_volatility_percentage_1


def normalize_data(data):
	# normalize open
	print(dt.datetime.now())
	data.loc[:, "open"] -= data.loc[:, "open"].mean(axis=0)
	data.loc[:, "open"] /= data.loc[:, "open"].std(axis=0)

	# print(data.loc[:, "open"].mean(axis=0))
	# print(data.loc[:, "open"].std(axis=0))

	#normalize high
	data.loc[:, "high"] -= data.loc[:, "high"].mean(axis=0)
	data.loc[:, "high"] /= data.loc[:, "high"].std(axis=0)
	# print(data.loc[:, "high"].mean(axis=0))
	# print(data.loc[:, "high"].std(axis=0))

	#normalize low
	data.loc[:, "low"] -= data.loc[:, "low"].mean(axis=0)
	data.loc[:, "low"] /= data.loc[:, "low"].std(axis=0)
	# print(data.loc[:, "low"].mean(axis=0))
	# print(data.loc[:, "low"].std(axis=0))

	#normalize close
	data.loc[:, "close"] -= data.loc[:, "close"].mean(axis=0)
	data.loc[:, "close"] /= data.loc[:, "close"].std(axis=0)
	# print(data.loc[:, "close"].mean(axis=0))
	# print(data.loc[:, "close"].std(axis=0))

	#normalize volume
	data.loc[:, "volume"] -= data.loc[:, "volume"].mean(axis=0)
	data.loc[:, "volume"] /= data.loc[:, "volume"].std(axis=0)
	# print(data.loc[:, "volume"].mean(axis=0))
	# print(data.loc[:, "volume"].std(axis=0))
	print(dt.datetime.now())

	#input(data)
	return data


pd.set_option('display.max_rows', None)


dsu_table = get_DSU_table()
recorded_tickers = dsu_table.Ticker.unique()
num_recorded_tickers = len(dsu_table)

for ticker in range(len(recorded_tickers)):
	current_ticker = recorded_tickers[ticker]
	print(current_ticker)
	current_tickers_dsu = dsu_table.loc[dsu_table['Ticker'] == current_ticker]
	current_tickers_dsu = current_tickers_dsu.reset_index(drop=True)
	# input(current_tickers_dsu)	
	all_current_tickers_data = get_tickers_data(current_ticker)

	for date_range_row in range(len(current_tickers_dsu)):
		current_start_date = current_tickers_dsu['start_date'][date_range_row]
		current_end_date = current_tickers_dsu['end_date'][date_range_row]
		
		date_ranges_ticker_data = all_current_tickers_data.loc[all_current_tickers_data['date'] >= current_start_date]
		date_ranges_ticker_data = date_ranges_ticker_data.loc[date_ranges_ticker_data['date']<=current_end_date]
		
		current_analysis_data = date_ranges_ticker_data.loc[date_ranges_ticker_data['date'] < current_end_date]
		current_trade_day_data = date_ranges_ticker_data.loc[date_ranges_ticker_data['date'] == current_end_date]
		
		analysis_data_mean_close = current_analysis_data.close.mean()
		analysis_data_mean_volume = current_analysis_data.volume.mean()
		analysis_data_close_std = current_analysis_data.close.std()

		analysis_dates = current_analysis_data.date.unique()
		analysis_trainData_df = pd.DataFrame(columns=["open_vs_avg_close_N", "high_vs_avg_close_N", "low_vs_avg_close_N", \
			"close_vs_avg_close_N", "vol_vs_avg_vol_N", "perc_volatility", "avg_30m_perc_volatility"]) 

		current_analysis_data = fill_trading_data.fill_analysis_data_v1(current_analysis_data)
		#normalized_analysis_data = normalize_data(current_analysis_data)

		'''
		OKAY, FOR FIRST ATTEMPT, SKIP INCLUDING ANALYSIS ROWS IN DATA, AND JUST NORMALIZE THE TRADE_DAY DATA
		USING THE ANALYSIS DATA
		'''
		'''
		for date_index in range(len(analysis_dates)):
			current_date = analysis_dates[date_index]
			dates_ticker_data = (current_analysis_data.loc[current_analysis_data['date']==current_date]).reset_index(drop=True)
			# input(dates_ticker_data)
			# dates_ticker_data = fill_trading_data.fill_trade_day_training_data_v1(dates_ticker_data)
			# input(dates_ticker_data)
			# dates_ticker_data_normalized = dates_ticker_data.loc[:,["open", "high", "low", "close", "volume"]]			

			dates_mean_open = dates_ticker_data.open.mean()
			dates_mean_low = dates_ticker_data.low.mean()
			dates_mean_high = dates_ticker_data.high.mean()
			dates_mean_close = dates_ticker_data.close.mean()
			dates_mean_volume = dates_ticker_data.volume.mean()
			dates_close_std = dates_ticker_data.close.std()

			dates_final_close = dates_ticker_data['close'].iloc[-1]		

			dates_open_vs_avg_close	= (dates_mean_open-analysis_data_mean_close)/analysis_data_mean_close
			dates_low_vs_avg_close = (dates_mean_low-analysis_data_mean_close)/analysis_data_mean_close
			dates_high_vs_avg_close = (dates_mean_high-analysis_data_mean_close)/analysis_data_mean_close
			dates_close_vs_avg_close = (dates_mean_close-analysis_data_mean_close)/analysis_data_mean_close
			dates_volume_vs_avg_volume = (dates_mean_volume-analysis_data_mean_volume)/analysis_data_mean_volume
			dates_percent_volatility = (2*dates_close_std)/dates_final_close
			dates_avg_30m_volatility = get_avg_30m_volatility(dates_ticker_data)

			analysis_trainData_df = analysis_trainData_df.append({"open_vs_avg_close_N": dates_open_vs_avg_close, \
				"high_vs_avg_close_N": dates_high_vs_avg_close, "low_vs_avg_close_N": dates_low_vs_avg_close, \
				"close_vs_avg_close_N": dates_close_vs_avg_close, "vol_vs_avg_vol_N": dates_volume_vs_avg_volume, \
				"perc_volatility": dates_percent_volatility, "avg_30m_perc_volatility":dates_avg_30m_volatility}, ignore_index=True)
		
		#print(analysis_trainData_df)
		'''


		trade_month_normalized = current_trade_day_data.date.iloc[0].month / 12
		#print(trade_month_normalized)

		trade_day_normalized = current_trade_day_data.date.iloc[0].day / 31
		#print(trade_day_normalized)

		trade_weekday_normalized = (current_trade_day_data.date.iloc[0].weekday() + 1) / 5
		#print(trade_weekday_normalized)

		train_data_time_row = pd.DataFrame([[trade_month_normalized, trade_day_normalized, \
			trade_weekday_normalized, 0, 0, 0, 0]], columns=["month_norm", "day_norm", "weekday_norm", "hour_norm", \
				"minute_norm", "empty_col", "empty_col"])
		#print(train_data_time_row)

		#means_values = pd.DataFrame([[analysis_data_mean_close, analysis_data_mean_volume]], columns=["mean_close", "mean_volume"])

		#training_data = pd.concat([analysis_trainData_df, train_data_time_row], axis=0).reset_index(drop=True)
		#input(analysis_trainData_df)
		#input(train_data_time_row)
		#input(means_values)


		trade_day_training_data_raw = current_trade_day_data.drop(['trading_day'], axis=1).reset_index(drop=True)
		trade_day_training_data = fill_trading_data.fill_trade_day_training_data_v1(trade_day_training_data_raw)

		'''
		trade_day_training_data["open"] = (trade_day_training_data["open"]-analysis_data_mean_close)/analysis_data_mean_close
		trade_day_training_data["high"] = (trade_day_training_data["high"]-analysis_data_mean_close)/analysis_data_mean_close
		trade_day_training_data["low"] = (trade_day_training_data["low"]-analysis_data_mean_close)/analysis_data_mean_close
		trade_day_training_data["close"] = (trade_day_training_data["close"]-analysis_data_mean_close)/analysis_data_mean_close
		trade_day_training_data["volume"] = (trade_day_training_data["volume"]-analysis_data_mean_volume)/analysis_data_mean_volume
		'''

		input(trade_day_training_data)
		for num_rows in range(len(trade_day_training_data)):
			#input(trade_day_training_data.head(num_rows+1))
			non_norm_trade_rows = trade_day_training_data.head(num_rows+1)

			analysis_plus_non_norm = pd.concat([current_analysis_data, non_norm_trade_rows], axis=0)
			normed_data = normalize_data(analysis_plus_non_norm)
			normed_trade_rows = normed_data.tail(num_rows+1)
			input(normed_trade_rows)

		print(current_end_date)
		






