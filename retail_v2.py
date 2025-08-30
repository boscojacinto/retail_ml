import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Dataset from https://statso.io/2024/08/19/price-elasticity-of-demand-case-study/
df = pd.read_csv('Competition_Data.csv')

print(df)
#    Index Fiscal_Week_ID   Store_ID   Item_ID   Price  Item_Quantity	Sales_Amount_No_Discount	Sales_Amount 	Sales_Amount
# 0      0        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 		206.44
# 1      1        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 		158.01
# 2      2        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 		278.03
# 3      3        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 		222.66
# 4      4        2019-11  store_459  item_526  134.49            435 					4716.74 		11272.59 		195.32

def calculate_ped():
	# PED = % change in demand / % change in price

	# Percentage change in price and quantity
	df['Price_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Price'].pct_change()
	df['Quantity_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Item_Quantity'].pct_change()

	# print(f"\nPrice_Change:{df['Price_Change']}")
	# print(f"\nQuantity_Change:{df['Quantity_Change']}")

	# PED
	df['PED'] = df['Quantity_Change'] / df['Price_Change']

	df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
	df.dropna(subset=['PED'], inplace=True)

	# ped_summary = df['PED'].describe()

	# print(ped_summary)

#calculate_ped()

df['log_price'] = np.log1p(df['Price'])
#df['log_ped'] = np.log1p(df['PED'])
df['log_quantity'] = np.log1p(df['Item_Quantity'])

X = df[['log_price']]
y = df['log_quantity']

print(df)

# print(f"X:{X}")
# print(f"y:{y}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Done training..")

y_pred = model.predict(X_test)
print(f"y_pred:{y_pred}")
print(f"y_test:{y_test.tolist()[0:10]}")

print(f"mse:{mean_squared_error(y_test, y_pred)}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (log_quantity): {rmse:.4f}")
print(f"RMSE: {np.exp(rmse)}")

ped = model.coef_[0]
print(f"Estimated PED (from log_price coefficient): {ped:.4f}")

new_price = 134.0
log_new_price = np.log1p(new_price)
log_new_price_scaled = scaler.transform([[log_new_price]])
q1 = model.predict(log_new_price_scaled)[0]  # Predicted log(quantity) at price
delta_p = 0.01 * new_price  # 1% price increase
log_new_price2 = np.log1p(new_price + delta_p)
log_new_price2_scaled = scaler.transform([[log_new_price2]])
q2 = model.predict(log_new_price2_scaled)[0]  # Predicted log(quantity) at price + delta
numerical_ped = (q2 - q1) / (log_new_price2 - log_new_price)  # Approx. PED
print(f"Numerical PED for price ${new_price}: {numerical_ped:.4f}")


calculate_ped()
provided_ped = df['PED'].mean()  # Example; use test set PED for exact validation
print(f"Provided PED (mean): {provided_ped:.4f}")

