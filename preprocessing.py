# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("./penguins.csv")
df.head()

# %%
df['Species'].unique()

# %%
df.info()

# %%
df.describe()

# %%
print(df[df.isna()==False])

# %%
df.isna().sum()

# %%
def fill_nulls(df):
    mask = df['Species'] == 'Gentoo'
    df.loc[mask] = df.loc[mask].fillna(df.loc[mask].mean(numeric_only=True))

    mask = df['Species'] == 'Adelie'
    df.loc[mask] = df.loc[mask].fillna(df.loc[mask].mean(numeric_only=True))
    
    mask = df['Species'] == 'Chinstrap'
    df.loc[mask] = df.loc[mask].fillna(df.loc[mask].mean(numeric_only=True))
    
    return df




# %%
df = fill_nulls(df)

# %%
df.duplicated().sum()

# %%
df.head()

# %%
df['OriginLocation'].value_counts()

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['OriginLocation'] = encoder.fit_transform(df['OriginLocation'])

# %%
df['OriginLocation'].value_counts()

# %%
df.info()

# %% [markdown]
# ### run this cell once

# %%
import os
# Create folder if it doesn't exist
os.makedirs("processed_data", exist_ok=True)
# Save the DataFrame as CSV inside that folder
df.to_csv("processed_data/processed_data.csv", index=False)
print("Processed data saved successfully in 'processed_data/processed_data.csv'")


