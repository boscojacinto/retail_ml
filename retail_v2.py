import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Dataset from https://statso.io/2024/08/19/price-elasticity-of-demand-case-study/
df = pd.read_csv('Competition_Data.csv')

def clean_dataset(df):
	df = df.drop(['Index', 'Competition_Price'], axis=1)
	df = df.drop_duplicates(subset=[
		'Fiscal_Week_ID', 'Store_ID',
		'Item_ID', 'Price', 'Item_Quantity',
		'Sales_Amount_No_Discount', 'Sales_Amount']).reset_index(drop=True)
	return df

def group(df, store_id: str, item_id: str):
	per_store_item = df.groupby(['Store_ID', 'Item_ID'])
	for name, group in per_store_item:
	    if name == (store_id, item_id):
	    	print(f"Group: {name}")
	    	print(group)
	    	print()

# def plot():
# 	# df['Quantity_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Item_Quantity']	
# 	# plt.figure(figsize=(6, 4))
# 	# plt.scatter(df['Index'], df['Sales_Amount'], color='blue', s=100)
# 	# plt.title('Index vs. Sales_Amount')
# 	# plt.xlabel('Index')
# 	# plt.ylabel('Sales_Amount')
# 	# plt.grid(True)
# 	# plt.savefig('./index_vs_sales_amount.png')
# 	# plt.close()

#pd.set_option('display.max_columns', None)

df = clean_dataset(df)
print(df)
group(df, 'store_162', 'item_743')
#plot()

#    Index Fiscal_Week_ID   Store_ID   Item_ID   Price  Item_Quantity	Sales_Amount_No_Discount	Sales_Amount 	Competition_Price
# 0      0        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 			206.44
# 1      1        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 			158.01
# 2      2        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 			278.03
# 3      3        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 			222.66
# 4      4        2019-11  store_459  item_526  134.49            435 					4716.74 		11272.59 			195.32

def calculate_ped():
	# PED = % change in demand / % change in price

	# Percentage change in price and quantity
	df['Price_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Price'].pct_change()
	df['Quantity_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Item_Quantity'].pct_change()

	print(f"\nPrice_Change:{df['Price_Change'].describe()}")
	print(f"\nQuantity_Change:{df['Quantity_Change'].describe()}")

	# PED
	df['PED'] = df['Quantity_Change'] / df['Price_Change']

	df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
	df.dropna(subset=['PED'], inplace=True)

	ped_summary = df['PED'].describe()

	print(ped_summary)

#calculate_ped()

def prepare_feature():
	# Encoding
	df['Store_ID'] = pd.factorize(df['Store_ID'])[0] + 1
	df['Item_ID'] = pd.factorize(df['Item_ID'])[0] + 1 
	
	# Conver to log scale
	df['log_price'] = np.log1p(df['Price'])
	df['log_quantity'] = np.log1p(df['Item_Quantity'])

	X = df[['log_price', 'Store_ID', 'Item_ID']]
	y = df['log_quantity']

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	# print(f"X_scaled:{X_scaled}")
	# print(f"X mean:{X_scaled.mean()}")
	# print(f"X std:{X_scaled.std()}")

	return X_scaled, y

# Prepare feature
X, y = prepare_feature()

# Split set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

print("Done training..")

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (log_quantity): {rmse:.4f}")

# PED
ped = model.coef_[0]
print(f"Estimated PED (from log_price coefficient): {ped:.4f}")
