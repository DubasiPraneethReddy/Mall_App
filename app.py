from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Mall_customers.csv")
encoded_df = df.copy()
encoded_df['Gender'] = encoded_df['Gender'].map({'Male': 0, 'Female': 1})

# Train K-Means Model
X = encoded_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
encoded_df['Cluster'] = kmeans.labels_

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get("age"))
    income = int(request.form.get("income"))
    score = int(request.form.get("score"))
    input_data = [[age, income, score]]
    cluster = kmeans.predict(input_data)[0]
    return render_template("results.html", cluster=cluster)

@app.route('/dataset/raw')
def dataset_raw():
    return df.to_html(classes="table")

@app.route('/dataset/encoded')
def dataset_encoded():
    return encoded_df.to_html(classes="table")

@app.route('/cluster-map')
def cluster_map():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=encoded_df['Annual Income (k$)'], y=encoded_df['Spending Score (1-100)'], hue=encoded_df['Cluster'], palette="viridis")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Clusters")
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/elbow-method')
def elbow_method():
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bo-', markersize=8)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


