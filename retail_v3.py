import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Private dataset
# 	 Product title	Product vendor	Product type	Net items sold	Gross sales	Discounts	Returns	Net sales	Taxes	Total sales
product = pd.read_csv('products_catalogue.csv')

# Private dataset
# 	 Product title Product vendor Product type  Net items sold  Gross sales		Discounts 	Returns  Net sales    Taxes  Total sales
sales = pd.read_csv('total_monthly_sales_by_product.csv')

# Clean Product Catalogue
def clean_product_catalogue(df):
	# Drop irrelevant columns
	df = df[['Handle', 'Title', 'Variant SKU', 'Variant Price', 'Variant Compare At Price']]

	# Drop rows with NaN in 'Variant Price' and 'Variant Compare At Price'
	df = df.dropna(subset=['Variant Price', 'Variant Compare At Price'])

	# Fill missing 'Title' 
	df.groupby(['Handle'])['Title'].transform(lambda x: x.fillna(x.iloc[0]))

	# Drop 'Handle'
	df = df.drop(['Handle'], axis=1) 	
	
	# Drop duplicates in 'Variant SKU'
	df = df.drop_duplicates(subset=['Variant SKU'], keep='first')

	return df

# Utility function to group by subset
def group(df, **kwargs):
	k_subset = []
	v_subset = []
	for k, v in kwargs.items():
		k_subset.append(k)
		v_subset.append(v)

	item_per_tuple = df.groupby(k_subset, sort=False)
	for name, group in item_per_tuple:
		if name == tuple(v_subset):
			return group

	return None

product = clean_product_catalogue(product)
print(f"product:\n{product}")

print(f"Group:\n{group(df=product, Title='Trisha Siva Kurta')}")
