import pandas as pd
import datetime as dt

from datetime import timedelta

def fill_intial_data_rows(original_data):
	const_start_time = dt.time(9,30,0)

	initial_rows_DF = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
	days_date = original_data.date[0]

	orig_data_start_time = original_data.time.iloc[0]
	orig_data_init_open = original_data.open.iloc[0]
	orig_data_init_high = original_data.high.iloc[0]
	orig_data_init_low = original_data.low.iloc[0]
	orig_data_init_close = original_data.close.iloc[0]

	if(orig_data_start_time > const_start_time):
		#print("Data Start Time is not 9:30:00")
		initial_time_DF = pd.DataFrame(columns=["date", "time"])
		df_row_counter = 0
		orig_data_init_volume = 0

		curr_start_time = dt.datetime.combine(days_date, const_start_time)
		while(curr_start_time.time() != orig_data_start_time):
			initial_time_DF.loc[df_row_counter] = [curr_start_time.date(), curr_start_time.time()]

			df_row_counter += 1
			curr_start_time = curr_start_time + timedelta(minutes=1)

		empty_minutes = len(initial_time_DF)
		orig_data_init_row = {"open":orig_data_init_open, "high":orig_data_init_high, "low":orig_data_init_low, "close":orig_data_init_close, "volume":orig_data_init_volume}
		
		initial_rows_DF = initial_rows_DF.append([orig_data_init_row]*empty_minutes, ignore_index=True)
		initial_rows_DF = pd.concat([initial_time_DF, initial_rows_DF], axis=1)			

	else:
		return#print("Data Start Time is 9:30:00")	

	return initial_rows_DF


def fill_final_data_rows(original_data):
	const_end_time = dt.time(16,0,0)

	final_rows_DF = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
	days_date = original_data.date[0]

	orig_data_end_time = original_data.time.iloc[-1]
	orig_data_final_open = original_data.open.iloc[-1]
	orig_data_final_high = original_data.high.iloc[-1]
	orig_data_final_low = original_data.low.iloc[-1]
	orig_data_final_close = original_data.close.iloc[-1]	

	if(orig_data_end_time < const_end_time):
		#print("Data End Time is not 16:00:00")
		final_time_DF = pd.DataFrame(columns=["date", "time"])
		df_row_counter = 0
		orig_data_final_volume = 0

		curr_end_time = dt.datetime.combine(days_date, orig_data_end_time) + timedelta(minutes=1)
		while(curr_end_time.time() != const_end_time):
			final_time_DF.loc[df_row_counter] = [curr_end_time.date(), curr_end_time.time()]

			df_row_counter += 1
			curr_end_time = curr_end_time + timedelta(minutes=1)

		final_empty_minutes = len(final_time_DF)
		orig_data_final_row = {"open":orig_data_final_open, "high":orig_data_final_high, "low":orig_data_final_low, "close":orig_data_final_close, "volume":orig_data_final_volume}
		
		final_rows_DF = final_rows_DF.append([orig_data_final_row]*final_empty_minutes, ignore_index=True)
		final_rows_DF = pd.concat([final_time_DF, final_rows_DF], axis=1)			

	else:
		return#print("Data End Time is 16:00:00")

	return final_rows_DF



def fill_middle_data_rows(original_data):
	days_date = original_data.date[0]

	filled_data_DF = pd.DataFrame(columns=["date", "time", "open", "high", "low", "close", "volume"])
	for row_index in range(len(original_data)):

		curr_data_row = original_data.loc[[row_index]]
		curr_data_row_datetime = dt.datetime.combine(curr_data_row["date"].iloc[0], curr_data_row["time"].iloc[0])

		if(row_index != len(original_data)-1):
			next_data_row = original_data.loc[[row_index+1]]
			next_data_row_datetime = dt.datetime.combine(next_data_row["date"].iloc[0], next_data_row["time"].iloc[0])

		time_difference = int((next_data_row_datetime - curr_data_row_datetime).seconds/60)

		proper_next_data_row_datetime = curr_data_row_datetime + timedelta(minutes=1)

		if(proper_next_data_row_datetime != next_data_row_datetime):
			filled_data_DF = filled_data_DF.append(curr_data_row)

			for minutes in range(time_difference-1):
				if(row_index < len(original_data)-1):
					prop_next_date = proper_next_data_row_datetime.date()
					prop_next_time = (proper_next_data_row_datetime + timedelta(minutes=minutes)).time()
					prop_next_open = curr_data_row.open.iloc[0]
					prop_next_high = curr_data_row.high.iloc[0]
					prop_next_low = curr_data_row.low.iloc[0]
					prop_next_close = curr_data_row.close.iloc[0]
					prop_next_volume = 0

					prop_next_row = {"date":prop_next_date, "time":prop_next_time, "open":prop_next_open, "high":prop_next_high, "low":prop_next_low, "close":prop_next_close, "volume":prop_next_volume}
					filled_data_DF = filled_data_DF.append([prop_next_row], ignore_index = True)

		else:
			filled_data_DF = filled_data_DF.append(curr_data_row)
	filled_data_DF = filled_data_DF.reset_index(drop=True)
	return filled_data_DF




def fill_trade_day_training_data_v1(original_data):
	trade_day_training_data = pd.DataFrame(columns=["date", "time", "open", "high", "low", "close", "volume"])

	#input(original_data)	
	
	days_date = original_data.date[0]

	initial_rows_DF = fill_intial_data_rows(original_data)
	filled_middle_data_DF = fill_middle_data_rows(original_data)
	final_rows_DF = fill_final_data_rows(original_data)	

	trade_day_training_data = pd.concat([initial_rows_DF, filled_middle_data_DF, final_rows_DF])
	trade_day_training_data = trade_day_training_data.reset_index(drop=True)

	return trade_day_training_data


def fill_analysis_data_v1(original_data):
	analysis_dates = original_data.date.unique()

	original_data = original_data.drop("trading_day", axis=1)

	new_data = pd.DataFrame(columns=["date", "time", "open", "high", "low", "close", "volume"])

	for date_index in range(len(analysis_dates)):
		current_date = analysis_dates[date_index]
		dates_ticker_data = (original_data.loc[original_data['date']==current_date]).reset_index(drop=True)

		initial_rows = fill_intial_data_rows(dates_ticker_data)
		middle_rows = fill_middle_data_rows(dates_ticker_data)
		final_rows = fill_final_data_rows(dates_ticker_data)

		rows_combined = pd.concat([initial_rows, middle_rows, final_rows])
		#input(rows_combined)

		new_data = new_data.append(rows_combined)

	new_data = new_data.reset_index(drop=True)
	#input(new_data)

	return new_data


