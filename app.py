#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/render')
def render_page():
    return render_template('render.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    category = data.get("category")

    # Dummy classification logic (replace with ML model if needed)
    result = f"Classification result for {category}: Success"
    
    return jsonify({"message": result})

if __name__ == '__main__':
    app.run(debug=True)

