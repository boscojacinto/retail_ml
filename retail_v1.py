import pandas as pd

# Import Kaggle dataset https://www.kaggle.com/datasets/saibattula/retail-price-dataset-sales-data
df = pd.read_csv('data.csv') #[84000 rows x 13 columns]

# Remove missing values
df = df.dropna() #[168000 rows x 13 columns]

# Superfical discount based on price tier
def apply_discount(price_tier):
	if price_tier == 'high':
		return 10
	elif price_tier == 'middle':
		return 5
	elif price_tier == 'low':
		return 2

# Superfical promo based on price tier
def apply_promo(price_tier):
	if price_tier == 'high':
		return "sale_10%"
	elif price_tier == 'middle':
		return "sale_5%"
	elif price_tier == 'low':
		return "sale_2%"

# Apply discount
df['discount'] = df['price_tier'].apply(apply_discount)

# Apply promo
df['promo'] = df['price_tier'].apply(apply_promo)

# Add inventory status
df['inventory_status'] = "available"

pd.set_option('display.max_columns', None)
print(df[1060:1080])

# print(f"Columns:{df.columns}")
# print(f"Index:{df.index}")