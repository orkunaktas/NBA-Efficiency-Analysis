# # NBA 2023/24 Efficiency Score

# In[27]:


import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")


# In[29]:


df = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2024_per_game.html', attrs={'id': 'per_game_stats'})[0]


# In[31]:


df.head()


# In[33]:


import pandas as pd

# Tekrar eden satırları bulma
duplicates = df[df.duplicated()]

# Tekrar eden satırları yazdırma
print("Tekrar Eden Satırlar:")
print(duplicates)


# In[35]:


df.duplicated().sum()


# In[37]:


df = df.drop_duplicates()


# In[39]:


df.duplicated().sum()


# In[41]:


df.columns


# In[43]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define weights
alpha = 1.0  # Points (PTS)
beta = 1.0   # Assists (AST)
gamma = 1.0  # Rebounds (TRB)
delta = 1.2  # Steals (STL)
epsilon = 1.1  # Blocks (BLK)
zeta = 0.2   # Turnovers (TOV)
eta = 0.6    # Field Goal % (FG%)
theta = 0.9  # Free Throw % (FT%)

# Ensure numeric data
numeric_columns = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV', 'FG%', 'FT%']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Calculate Efficiency Score
df['Efficiency_Score'] = (alpha * df['PTS'] +
                          beta * df['AST'] +
                          gamma * df['TRB'] +
                          delta * df['STL'] +
                          epsilon * df['BLK'] -
                          zeta * df['TOV'] +
                          eta * df['FG%'] +
                          theta * df['FT%'])

# Drop rows with NaN values in 'Efficiency_Score'
df = df.dropna(subset=['Efficiency_Score'])

# Get the top 20 players by Efficiency Score
top_20_players = df[['Player', 'Pos', 'Efficiency_Score']].sort_values(by='Efficiency_Score', ascending=False).head(20)

# Plotting
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

bar_plot = sns.barplot(x='Efficiency_Score', y='Player', data=top_20_players, edgecolor='black')

plt.xlabel('Efficiency Score')
plt.ylabel('Player')
plt.title('Top 20 NBA Players by Efficiency Score')

plt.show()


# In[45]:


all = df[(df["Efficiency_Score"] >= 36)].sort_values(by="Efficiency_Score",ascending=False).reset_index(drop=True)


# In[47]:


all[["Player","Pos","Age","Tm","Efficiency_Score"]]


# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Define weights
alpha = 1.0  # Points (PTS)
beta = 1.0   # Assists (AST)
gamma = 1.0  # Rebounds (TRB)
delta = 1.2  # Steals (STL)
epsilon = 1.1  # Blocks (BLK)
zeta = 0.2   # Turnovers (TOV)
eta = 0.6    # Field Goal % (FG%)
theta = 0.9  # Free Throw % (FT%)

# Ensure numeric data
numeric_columns = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV', 'FG%', 'FT%']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Calculate Efficiency Score
df['Efficiency_Score'] = (alpha * df['PTS'] +
                          beta * df['AST'] +
                          gamma * df['TRB'] +
                          delta * df['STL'] +
                          epsilon * df['BLK'] -
                          zeta * df['TOV'] +
                          eta * df['FG%'] +
                          theta * df['FT%'])

# Drop rows with NaN values in 'Efficiency_Score'
df = df.dropna(subset=['Efficiency_Score'])

# Define features and target
features = df[['PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV', 'FG%', 'FT%']]
target = df['Efficiency_Score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Optionally, you can inspect the feature importances
importances = model.feature_importances_
features_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
print(features_importance_df.sort_values(by='Importance', ascending=False))


# In[51]:


y_pred


# In[53]:


test_data = X_test.copy()
test_data['Actual_Score'] = y_test
test_data['Predicted_Score'] = y_pred

test_data = pd.concat([test_data, df.loc[X_test.index, ['Player']]], axis=1)

print(test_data[['Player', 'Actual_Score', 'Predicted_Score']])


# In[55]:


top_10_actual_scores = test_data[['Player', 'Actual_Score',"Predicted_Score"]].sort_values(by='Actual_Score', ascending=False).head(20)

print("En Yüksek Gerçek Verimlilik Skorlarına Sahip 10 Oyuncu:")
print(top_10_actual_scores.to_string(index=False))


# In[57]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs. Tahmin Edilen Değerler')
plt.show()


# In[ ]:




