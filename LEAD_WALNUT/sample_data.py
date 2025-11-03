import pandas as pd

# Load and sample the dataset
df = pd.read_csv('data (1).csv')
sampled = df.sample(n=65, random_state=42)
sampled.to_csv('data/data.csv', index=False)

print(f'Sampled {len(sampled)} rows')
print(f'Columns: {list(sampled.columns)}')
print(f'Sample shape: {sampled.shape}')
