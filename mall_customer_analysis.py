#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("Mall_customers.csv")
data


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)


# In[5]:


data.dtypes


# In[6]:


print(data.head())


# In[7]:


print(data.isnull().sum())


# In[8]:


print(data.describe())


# In[9]:


print(data['Gender'].value_counts())


# In[10]:


# Summary statistics for Age
print(data['Age'].describe())


# In[11]:


plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[12]:


# Boxplot of Age
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Age'])
plt.title('Boxplot of Age')
plt.show()


# In[13]:


# Summary statistics of Annual Spend
print(data['Annual Income (k$)'].describe())


# In[14]:


import matplotlib.pyplot as plt
# Histogram for Income
plt.hist(data['Annual Income (k$)'], bins=20, color='red', edgecolor='black')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income')
plt.ylabel('Frequency')
plt.show()


# In[15]:


# Boxplot of Annual Income
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Annual Income (k$)'])
plt.title('Boxplot of Annual Income')
plt.show()


# In[16]:


# Summary statistics of Annual Spend
print(data['Spending Score (1-100)'].describe())


# In[17]:


# Histogram for Spending
plt.hist(data['Spending Score (1-100)'], bins=20, color='orange', edgecolor='black')
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score')
plt.ylabel('Frequency')
plt.show()


# In[18]:


# Boxplot for spending
sns.boxplot(x=data['Spending Score (1-100)'], color='blue')
plt.title('Spending Score Distribution')
plt.show()


# In[19]:


# Summary statistics of Annual Spend
print(data['Annual_Spend'].describe())


# In[20]:


plt.figure(figsize=(8, 6))
sns.histplot(data['Annual_Spend'], kde=True, bins=30)
plt.title('Distribution of Annual Spend')
plt.show()


# In[21]:


# Boxplot of Annual Spend
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Annual_Spend'])
plt.title('Boxplot of Annual Spend')
plt.show()


# In[22]:


# Summary statistics of Annual Spend
print(data['Visit_Frequency'].describe())


# In[23]:


data['Visit_Frequency'] = pd.to_datetime(data['Visit_Frequency'])
data['Month'] = data['Visit_Frequency'].dt.month
data['DayOfWeek'] = data['Visit_Frequency'].dt.dayofweek


# In[24]:


sns.countplot(x='Month', data=data, color='orange')
plt.title('Customer Visits by Month')
plt.show()


# In[25]:


sns.countplot(x='DayOfWeek', data=data, color='lightgreen' )
plt.title('Customer Visits by Day of the Week')
plt.show()


# In[26]:


corr_matrix = data.select_dtypes(include=['number']).corr()
print(corr_matrix)


# In[27]:


from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])  # Converts 'Male'/'Female' to 0/1

# Now compute correlation
corr_matrix = data.corr()


# In[28]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

print("Before Encoding:", data['Gender'].unique())

data['Gender'] = encoder.fit_transform(data['Gender'])

print("After Encoding:", data['Gender'].unique())


# In[29]:


corr_matrix = data.corr()
print(corr_matrix)


# In[30]:


from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
encoder = LabelEncoder()
data['Month'] = encoder.fit_transform(data['Month'])
data['DayOfWeek'] = encoder.fit_transform(data['DayOfWeek'])

# Recalculate correlation
corr_matrix = data.corr()
print(corr_matrix)


# In[31]:


data = data.drop(columns=['CustomerID'])


# In[32]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=['number'])), columns=data.select_dtypes(include=['number']).columns)

# Compute correlation again
corr_matrix = data_scaled.corr()
print(corr_matrix)


# In[33]:


# Ensure 'Month' and 'DayOfWeek' are converted to numeric
data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
data['DayOfWeek'] = pd.to_numeric(data['DayOfWeek'], errors='coerce')

# Check for missing values
print(data[['Month', 'DayOfWeek']].isnull().sum())  # See if NaNs exist

# Fill missing values with median (or another strategy)
data['Month'].fillna(data['Month'].median(), inplace=True)
data['DayOfWeek'].fillna(data['DayOfWeek'].median(), inplace=True)

# Compute correlation again
corr_matrix = data.corr()
print(corr_matrix)


# In[34]:


data['Month'] = data['Month'].astype(int)
data['DayOfWeek'] = data['DayOfWeek'].astype(int)

# Compute correlation again
corr_matrix = data.corr()
print(corr_matrix)


# In[35]:


data['Month'] = data['Month'].astype(int)
data['DayOfWeek'] = data['DayOfWeek'].astype(int)

# Compute correlation again
corr_matrix = data.corr()
print(corr_matrix)


# In[36]:


# Check the data types again
print(data.dtypes)

# Convert 'Month' and 'DayOfWeek' to numeric
data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
data['DayOfWeek'] = pd.to_numeric(data['DayOfWeek'], errors='coerce')

# Check for NaN values after conversion
print(data[['Month', 'DayOfWeek']].isnull().sum())

# Fill NaNs if any exist (optional)
data['Month'].fillna(data['Month'].median(), inplace=True)
data['DayOfWeek'].fillna(data['DayOfWeek'].median(), inplace=True)

# Compute correlation again
corr_matrix = data.corr()
print(corr_matrix)


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[38]:


data.isnull()


# In[39]:


data.describe()


# In[40]:


data1 = data.iloc[:,1:]


# In[41]:


data1


# In[42]:


cols = data.columns


# In[43]:


data.info()


# In[44]:


data1 = pd.get_dummies(data1, drop_first=True)


# In[45]:


cols = data1.columns


# In[46]:


feature_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
features = data1[feature_columns]


# In[47]:


kmeans = KMeans(n_clusters=5, random_state=42)
data1['Cluster'] = kmeans.fit_predict(features)


# In[49]:


print(data1['Cluster'].value_counts())


# In[50]:


labels = kmeans.labels_
print(labels)  # This will print the cluster labels array


# In[52]:


data[data1['Cluster']==0]


# In[53]:


print(data1[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head(10))


# In[54]:


X = data1.drop(columns=['Visit_Frequency'])  
y = data1['Visit_Frequency']


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[57]:


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)


# In[58]:


st.title(" Mall Customers Analysis")
st.subheader(" Model Performance")


# In[59]:


print(y_test.dtypes)


# In[60]:


y_test = y_test.astype('int64')  # Converts datetime to int (nanoseconds since epoch)


# In[61]:


print(X.dtypes)  # Check data types of all columns
print(X.head())  # Look at the first few rows


# In[63]:


print(X.dtypes)  


# In[66]:


print(X.dtypes)


# In[67]:


print(y.dtypes)  


# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

st.title("Mall Customers Analysis")
st.subheader("Exploratory Data Analysis & Clustering")

feature_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
st.write("### Select Features for Clustering:")
selected_features = st.multiselect("Choose features", feature_columns, default=feature_columns)

if len(selected_features) >= 2:
    features = data[selected_features]
    

    kmeans = KMeans(n_clusters=5, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features)
    
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(data1[selected_features[0]], data1[selected_features[1]], c=data1['Cluster'], cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    ax.set_title('Customer Segments')
    ax.legend()
    st.pyplot(fig)
    
    st.write("### Cluster Counts:")
    st.write(data1['Cluster'].value_counts())

st.subheader("Model Performance")
X = data1.drop(columns=['Cluster', 'Visit_Frequency'])
y = data1['Spending Score (1-100)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

mse_lin = mean_squared_error(y_test, lin_reg.predict(X_test))
mse_tree = mean_squared_error(y_test, tree_reg.predict(X_test))

st.write(f"**Linear Regression MSE:** {mse_lin:.2f}")
st.write(f"**Decision Tree Regression MSE:** {mse_tree:.2f}")
st.subheader("Decision Tree Visualization")
buffer = io.BytesIO()
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(tree_reg, filled=True, feature_names=X.columns, ax=ax)
plt.savefig(buffer, format="png")
st.image(buffer, caption="Decision Tree Structure")



# In[69]:


fig, ax = plt.subplots()
scatter = ax.scatter(data1[selected_features[0]], data1[selected_features[1]], c=data1['Cluster'], cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
ax.set_xlabel(selected_features[0])
ax.set_ylabel(selected_features[1])
ax.set_title('Customer Segments')
ax.legend()
st.pyplot(fig)


# In[72]:


fig, ax = plt.subplots()
scatter = ax.scatter(data1[selected_features[0]], data1[selected_features[1]], c=data1['Cluster'], cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')

ax.set_xlabel(selected_features[0])
ax.set_ylabel(selected_features[1])
ax.set_title('Customer Segments')

ax.legend()

plt.show()


# In[79]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "Mall_customers.csv"  # Ensure the file is in the same directory as the Jupyter Notebook
df = pd.read_csv(file_path)

feature_columns = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[feature_columns]



y = data1["Cluster"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
def predict_cluster(annual_income, spending_score):
    return rf_classifier.predict(np.array([[annual_income, spending_score]]))[0]

# Example usage
example_income = 40
example_spending = 60
predicted_cluster = predict_cluster(example_income, example_spending)
print(f"Predicted Cluster: {predicted_cluster}")



# In[ ]:




