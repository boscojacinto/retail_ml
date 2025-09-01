import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Dataset from https://statso.io/2024/08/19/price-elasticity-of-demand-case-study/
#    Index Fiscal_Week_ID   Store_ID   Item_ID   Price  Item_Quantity	Sales_Amount_No_Discount	Sales_Amount 	Competition_Price
# 0      0        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 			206.44
# 1      1        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 			158.01
# 2      2        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 			278.03
# 3      3        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 			222.66
# 4      4        2019-11  store_459  item_526  134.49            435 					4716.74 		11272.59 			195.32

df = pd.read_csv('Competition_Data.csv')

# Clean Dataset
def clean_dataset(df):
	# Drop Index
	df = df.drop(['Index'], axis=1)

	# Drop duplicates; Keep first entry competition price
	df = df.drop_duplicates(subset="Price")

	return df

# Utility function to group by Store Id and Item Id
def group(df, store_id: str, item_id: str):
	per_store_item = df.groupby(['Store_ID', 'Item_ID'])
	for name, group in per_store_item:
	    if name == (store_id, item_id):
	    	print(f"Group: {name}")
	    	print(group)
	    	return group

	return None

# Utility function to add seasonality
def get_seasonality(fiscal_week_id):
    week = int(fiscal_week_id.split('-')[1])
    
    if week in range(1, 9) or week in range(48, 53):
        return 'Winter'
    elif week in range(9, 22):
        return 'Spring'
    elif week in range(22, 35):
        return 'Summer'
    else:
        return 'Fall'

# Utility function to add category
def get_price_category(price):
    if price < 130:
        return 'Low'
    elif 130 <= price < 135:
        return 'Medium'
    else:  # price >= 135
        return 'High'

# Utility function to calculate ped
def calculate_ped(df):
	# PED = % change in demand / % change in price

	# Percentage change in price and quantity
	df['PED'] = df.groupby(['Store_ID', 'Item_ID'])['Item_Quantity'].pct_change() / \
					df.groupby(['Store_ID', 'Item_ID'])['Price'].pct_change()

	df['PED'] = df['PED'].replace(float('nan'), 0.0)

	ped_summary = df['PED'].describe()
	print(f"ped_summary:{ped_summary}")
	return df

# Prepare Dataset
def prepare_dataset(df):
	df['Seasonality'] = df['Fiscal_Week_ID'].apply(get_seasonality)
	df['Category'] = df['Price'].apply(get_price_category)
	df['Discount_Percentage'] = df.apply(
	    lambda row: ((row['Competition_Price'] - row['Price']) / row['Competition_Price'] * 100)
	    if row['Competition_Price'] > row['Price'] and row['Competition_Price'] > 0 else 0,
	    axis=1
	)
	df['Discount_Percentage'] = df['Discount_Percentage'].round(2)
	df['Sales_Amount'] = df.apply(
		lambda row: (row['Price'] * row['Item_Quantity']),
		axis=1
	)
	df = calculate_ped(df)
	return df

def plot(df):
	df['Quantity_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Item_Quantity'].pct_change()
	df['Price_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Price'].pct_change()	
	plt.figure(figsize=(6, 4))
	plt.scatter(df['Quantity_Change'], df['Price_Change'], color='blue', s=100)
	plt.title('Quantity_Change vs. Price_Change')
	plt.xlabel('Quantity_Change')
	plt.ylabel('Price_Change')
	plt.grid(True)
	plt.savefig('./quantity_change_vs_price_change.png')
	plt.close()

df = clean_dataset(df)
df = prepare_dataset(df)

#print(df)

#group(df, 'store_162', 'item_743')
plot(df)

def train_predict_eval_ped():
	def prepare_feature():
		global df

		# Convert to log scale
		df['log_price'] = np.log1p(df['Price'])
		df['log_quantity'] = np.log1p(df['Item_Quantity'])

		X = df[['log_price']]
		y = df['log_quantity']

		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		return X_scaled, y

	# Prepare feature
	X, y = prepare_feature()

	# Split set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train
	model = LinearRegression()
	model.fit(X_train, y_train)

	print("Done training ped..")

	# Predict
	y_pred = model.predict(X_test)

	# Evaluate
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	print(f"RMSE (log_quantity): {rmse:.4f}")

	# PED
	ped = model.coef_[0]
	print(f"Estimated PED (from log_price coefficient): {ped:.4f}")

train_predict_eval_ped()

def train_predict_eval_discount():
	def prepare_feature():
		global df

		# Convert to log scale
		df['numerical_category'] = pd.factorize(df['Category'])[0] + 1
		df['numerical_seasonality'] = pd.factorize(df['Seasonality'])[0] + 1

		X = df[['Price', 'Discount_Percentage', 'PED', 'numerical_category', 'numerical_seasonality']]
		y = df['Sales_Amount']

		return X, y

	# Prepare feature
	X, y = prepare_feature()

	# Split set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train
	model = RandomForestRegressor(n_estimators=100, random_state=69)
	model.fit(X_train, y_train)

	print("Done training discount..")

	def predict_optimal_discount(X_test, y_test, discount_range=[0, 0.1, 0.2, 0.3]):
	    max_revenue = y_test.iloc[0]
	    optimal_discount = 0
	    for discount in discount_range:
	        df['Discount_Range'] = discount
	        predicted_revenue = model.predict(X_test[['Price', 'Discount_Percentage', 'PED', 'numerical_category', 'numerical_seasonality']])[0]
	        print(f"discount:{discount}, optimal_discount:{optimal_discount}, predicted_revenue:{predicted_revenue:.2f}, max_revenue:{max_revenue:.2f}")
	        if predicted_revenue > max_revenue:
	            max_revenue = predicted_revenue
	            optimal_discount = discount
	    return optimal_discount, max_revenue

	# Predict discount for random test sample 
	max_index = len(X_test) - 1
	r_index = random.randint(0, max_index)
	X_test_predict = X_test.iloc[r_index-1:r_index].copy()
	y_test_predict = y_test.iloc[r_index-1:r_index].copy()

	optimal_discount, predicted_revenue = predict_optimal_discount(X_test_predict, y_test_predict)
	print(f"Optimal Discount: {optimal_discount*100}%, Predicted Revenue: {predicted_revenue}")

train_predict_eval_discount()