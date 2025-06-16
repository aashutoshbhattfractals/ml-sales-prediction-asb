from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


import os

# List all files in the current directory
current_dir = os.getcwd()
files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]

print("Files in the current directory:")
for file in files:
    print(file)


# Load the model
with open('./model.pkl', 'rb') as f:
  model = pickle.load(f)

@app.route('/')
def home():
  return jsonify({'message': 'Sales Prediction API is running'})

@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json()
  features = np.array(data['features']).reshape(1, -1)
  prediction = model.predict(features)
  return jsonify({'predicted_sales': prediction[0]})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)
