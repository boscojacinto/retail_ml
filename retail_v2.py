import pandas as pd

# Dataset from https://statso.io/2024/08/19/price-elasticity-of-demand-case-study/
df = pd.read_csv('Competition_Data.csv')

#print(df.head())
#    Index Fiscal_Week_ID   Store_ID   Item_ID   Price  Item_Quantity	Sales_Amount_No_Discount	Sales_Amount 	Sales_Amount
# 0      0        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 		206.44
# 1      1        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 		158.01
# 2      2        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 		278.03
# 3      3        2019-11  store_459  item_526  134.49            435   				4716.74 		11272.59 		222.66
# 4      4        2019-11  store_459  item_526  134.49            435 					4716.74 		11272.59 		195.32

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

ped_summary = df['PED'].describe()

print(ped_summary)
