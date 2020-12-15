
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



def get_DSU_table(shuffle=False):
	engine = make_engine("StockInfo")
	columns = []
	dsuDF = pd.DataFrame()

	get_distinct_stock_updates1 = "select max(id) id, Ticker, start_date, end_date, daily_avg_price, daily_avg_volume, avg_daily_high, "
	get_distinct_stock_updates2 = "avg_daily_low, avg_daily_volatility_percentage, num_data_points, avg_30m_volatility_percentage "
	get_distinct_stock_updates3 = "from DailyStockUpdate "
	get_distinct_stock_updates4 = "group by Ticker, start_date, end_date, daily_avg_price, daily_avg_volume, avg_daily_high, "
	get_distinct_stock_updates5 = "avg_daily_low, avg_daily_volatility_percentage, num_data_points, avg_30m_volatility_percentage " 
	get_distinct_stock_updates6 = "order by Ticker"

	get_distinct_stock_updates_QUERY = get_distinct_stock_updates1+get_distinct_stock_updates2+get_distinct_stock_updates3+get_distinct_stock_updates4+get_distinct_stock_updates5+get_distinct_stock_updates6

	with engine.connect() as connection:
		columns = connection.execute("SHOW columns FROM StockInfo.{}".format("DailyStockUpdate"))
		columns = [r[0] for r in columns]

	with engine.connect() as connection:
		dsuDF = pd.DataFrame(connection.execute(get_distinct_stock_updates_QUERY), columns=columns)

	dsuDF = dsuDF.drop('id', axis=1)

	if(shuffle==True):
		dsuDF = dsuDF.sample(frac=1).sample(frac=1).sample(frac=1).sample(frac=1).sample(frac=1).sample(frac=1)\
										.sample(frac=1).sample(frac=1).reset_index(drop=True)

	engine.dispose() 
	return(dsuDF)


def get_tickers_data(ticker, separate_dt=False, start=None, end=None):
	engine = make_engine("StockInfo")

	with engine.connect() as connection:
		columns = connection.execute("SHOW columns FROM StockInfo.{}".format(ticker))
		columns = [r[0] for r in columns]
		query = ("SELECT * FROM StockInfo.{}").format(ticker)
		if(start != None):
			query =  query + (" WHERE datetime >= '{}'").format(start)
			if(end!=None):
				end = end + timedelta(days=1)
				query = query + (" AND datetime <= '{}'").format(end)

		elif(end != None):
			end = end + timedelta(days=1)
			query = query + (" WHERE datetime <= '{}'").format(end)

		tickerDf = pd.DataFrame(connection.execute(query), columns=columns)
		tickerDf = tickerDf.drop('id', axis=1)

		if(separate_dt != False):
			tickerDf.datetime = tickerDf.datetime.astype(str)
			
			#converts datetime column to to datetime.datetime type\
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

		else:
			engine.dispose()			
			return tickerDf

def cudf_get_tickers_data(ticker):
	engine = make_engine("StockInfo")

	with engine.connect() as connection:
		columns = connection.execute("SHOW columns FROM StockInfo.{}".format(ticker))
		columns = [r[0] for r in columns]

		tickerDf = pd.DataFrame(connection.execute("SELECT * FROM StockInfo.{}".format(ticker)), columns=columns)
		tickerDf = cudf.DataFrame.from_pandas(tickerDf)
		tickerDf = tickerDf.drop('id', axis=1)

	engine.dispose()
	
	return tickerDf


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
	# avg_30m_volatility = ((sum30mSTD/num_periods)*2)/days_last_close
	# input("calc at end of day: " + str(avg_30m_volatility))

	return avg_30m_volatility_percentage_1


def normalize_data(data):
	# normalize open
	data.loc[:, "open"] -= data.loc[:, "open"].mean(axis=0)
	data.loc[:, "open"] /= data.loc[:, "open"].std(axis=0)

	#normalize high
	data.loc[:, "high"] -= data.loc[:, "high"].mean(axis=0)
	data.loc[:, "high"] /= data.loc[:, "high"].std(axis=0)

	#normalize low
	data.loc[:, "low"] -= data.loc[:, "low"].mean(axis=0)
	data.loc[:, "low"] /= data.loc[:, "low"].std(axis=0)

	#normalize close
	data.loc[:, "close"] -= data.loc[:, "close"].mean(axis=0)
	data.loc[:, "close"] /= data.loc[:, "close"].std(axis=0)

	#normalize volume
	data.loc[:, "volume"] -= data.loc[:, "volume"].mean(axis=0)
	data.loc[:, "volume"] /= data.loc[:, "volume"].std(axis=0)
	return data


pd.set_option('display.max_rows', None)

'''
*This training data will be created by filling both the analysis data and trade day data setting prices to the previous price and volume to 0 for 
*filled data points. The trade day data will then be stepped through minute by minute, added to the analysis data, then normalized with the 
*analysis data, afterwards it will be separated from the analysis data to create the next "step" in the environment.
*This will be done using cudf instead of pandas
'''
def create_training_data_v1():
	print(dt.datetime.now())

	dsu_table = get_DSU_table()
	recorded_tickers = dsu_table.Ticker.unique()
	num_recorded_tickers = len(dsu_table)

	for ticker in range(len(recorded_tickers)):
		current_ticker = recorded_tickers[ticker]
		print(current_ticker)

		current_tickers_dsu = dsu_table.loc[dsu_table['Ticker'] == current_ticker]
		current_tickers_dsu = current_tickers_dsu.reset_index(drop=True)

		#all_current_tickers_data = get_tickers_data(current_ticker)
		all_current_tickers_data = get_tickers_data(current_ticker)
		# input(all_current_tickers_data)

		for date_range_row in range(len(current_tickers_dsu)):
			current_start_date = current_tickers_dsu['start_date'][date_range_row]
			current_start_date = np.datetime64(current_start_date)

			current_end_date = current_tickers_dsu['end_date'][date_range_row]
			current_end_date = np.datetime64(current_end_date) #+ np.timedelta64(1, 'D')
			
			date_ranges_ticker_data = all_current_tickers_data.loc[all_current_tickers_data['datetime'] >= current_start_date]
			date_ranges_ticker_data = date_ranges_ticker_data.loc[date_ranges_ticker_data['datetime']<= (current_end_date+np.timedelta64(1, 'D'))]
			
			current_analysis_data = date_ranges_ticker_data.loc[date_ranges_ticker_data['datetime'] < current_end_date]
			current_trade_day_data = date_ranges_ticker_data.loc[date_ranges_ticker_data['datetime'] >= current_end_date]
			
			# analysis_data_mean_close = current_analysis_data.close.mean()
			# analysis_data_mean_volume = current_analysis_data.volume.mean()
			# analysis_data_close_std = current_analysis_data.close.std() 

			current_analysis_data = fill_trading_data.cudf_fill_analysis_data_v1(current_analysis_data)

			trade_month_normalized = current_trade_day_data.datetime.iloc[0].month / 12	
			trade_day_normalized = current_trade_day_data.datetime.iloc[0].day / 31
			trade_weekday_normalized = (current_trade_day_data.datetime.iloc[0].weekday() + 1) / 5

			train_data_time_row = cudf.DataFrame([[trade_month_normalized, trade_day_normalized, \
				trade_weekday_normalized, 0, 0]], columns=["month_norm", "day_norm", "weekday_norm", "hour_norm", "minute_norm"])
		

			trade_day_training_data_raw = current_trade_day_data.drop(['trading_day'], axis=1).reset_index(drop=True)
			trade_day_training_data = fill_trading_data.cudf_fill_trade_training_data_v1(trade_day_training_data_raw)

			trade_day_training_data = cudf.DataFrame.from_pandas(trade_day_training_data)
			current_analysis_data = cudf.DataFrame.from_pandas(current_analysis_data)
			print(dt.datetime.now())

			prices_and_volume_df = trade_day_training_data.drop(['datetime'], axis=1).reset_index(drop=True)

			training_data_csv_df = cudf.DataFrame()
			for num_rows in range(len(trade_day_training_data)):
				# print(dt.datetime.now())
				#input(trade_day_training_data.head(num_rows+1))
				step_df = cudf.DataFrame([str(num_rows+1)]*(num_rows+1),columns=["step"])

				non_norm_trade_rows = trade_day_training_data.head(num_rows+1)

				analysis_plus_non_norm = cudf.concat([current_analysis_data, non_norm_trade_rows], axis=0)
				normed_data = normalize_data(analysis_plus_non_norm)
				normed_trade_rows = normed_data.tail(num_rows+1).rename(columns={"open":"normed_open", "high":"normed_high", "low":"normed_low", \
																		"close":"normed_close", "volume":"normed_volume"})
				
				price_and_vol_rows = prices_and_volume_df.head(num_rows+1)

				training_data_step_rows = cudf.concat([normed_trade_rows, step_df], axis=1).reset_index(drop=True)
				training_data_csv_df = cudf.concat([training_data_csv_df, training_data_step_rows], axis=0).reset_index(drop=True)


			input(dt.datetime.now())
			input(training_data_csv_df)
			
			# fileName = str(training_data_csv_df.datetime.iloc[0])+".csv"
			# training_data_csv_df.to_csv(fileName, index=False)
			# input("CSV created")


			print(current_end_date)

			#analysis_dates = current_analysis_data.datetime.unique()
			# print(analysis_dates)
			# analysis_trainData_df = cudf.DataFrame(columns=["open_vs_avg_close_N", "high_vs_avg_close_N", "low_vs_avg_close_N", \
			# 	"close_vs_avg_close_N", "vol_vs_avg_vol_N", "perc_volatility", "avg_30m_perc_volatility"]) 

			# input(analysis_dates.dtype)
			# input(analysis_trainData_df)

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

def fill_ticker_daterange_dataset(date_ranges_ticker_data):
	current_end_date = np.datetime64(pd.to_datetime(date_ranges_ticker_data.loc[:, "datetime"]).dt.date.unique().tolist()[-1])

	current_analysis_data = date_ranges_ticker_data.loc[date_ranges_ticker_data['datetime'] < current_end_date]
	current_trade_day_data = date_ranges_ticker_data.loc[date_ranges_ticker_data['datetime'] >= current_end_date]

	analysis_data_filled = fill_trading_data.cudf_fill_analysis_data_v1(current_analysis_data)

	trade_day_training_data_raw = current_trade_day_data.drop(['trading_day'], axis=1).reset_index(drop=True)
	trade_day_data_filled = fill_trading_data.cudf_fill_trade_training_data_v1(trade_day_training_data_raw)

	return analysis_data_filled, trade_day_data_filled





dsu_table = get_DSU_table(shuffle=True)
# input(dsu_table)

print(dt.datetime.now())
for current_row in range(len(dsu_table)):
	rows_ticker = dsu_table.Ticker.iloc[current_row]
	rows_start_date = dsu_table.start_date.iloc[current_row]
	rows_end_date = dsu_table.end_date.iloc[current_row]
	print(rows_ticker)
	# print(rows_start_date)
	print(rows_end_date)


	ticker_ranges_data = get_tickers_data(ticker=rows_ticker, start=rows_start_date, end=rows_end_date)
	#input(ticker_ranges_data)
	analysis_data, trade_day_data = fill_ticker_daterange_dataset(ticker_ranges_data)
	# input(analysis_data)
	# input(trade_day_data)
	if(current_row == 99):
		break
print(dt.datetime.now())





# choose_rand_ticker_and_date(dsu_table)

# create_training_data_v1()




