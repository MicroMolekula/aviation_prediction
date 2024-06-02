from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import re
import math

app = Flask(__name__)

'''
{
    "airline":
    "dep_time":
    "from":
    "time_taken":
    "stop":
    "arr_time":
    "to":
    "class":
}
'''

# Загрузка модели
model = load_model('model.h5')

def time_to_inteval(value):
    interval_value = 4
    minutes = int(value.split(":")[0])
    interval = math.floor(minutes / interval_value)
    return interval

def norm(value, max_value, min_value):
    return (value-min_value)/(max_value-min_value)

def airline_normalize(value):
    airlines = ['GO FIRST', 'Air India', 'Indigo', 'Vistara']
    result = airlines.index(value)
    return norm(result, 3, 0)

def from_to_normalize(value):
    cities = ['Bangalore', 'Delhi', 'Hyderabad', 'Kolkata', 'Chennai', 'Mumbai']
    result = cities.index(value)
    return norm(result, 5, 0)

def dep_arr_time_normalize(value):
    result = time_to_inteval(value)
    return norm(result, 5, 0)

def time_taken_normalize(value):
    hours_match = re.search(r'(\d+)h', value)
    minutes_match = re.search(r'(\d+)m', value)
    hours = int(hours_match.group(1)) if hours_match else 0
    minutes = int(minutes_match.group(1)) if minutes_match else 0
    result = math.floor((hours + minutes/60)/6)
    return norm(result, 8, 0)

def stop_normalize(value):
    stops = ['non-stop', '1-stop', '2+-stop']
    result = stops.index(value)
    return norm(result, 2, 0)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        result = []
        data = request.json
        data = dict(data)
        for row in data.values():
            row = [
                    dep_arr_time_normalize(row["dep_time"]), 
                    time_taken_normalize(row["time_taken"]), 
                    airline_normalize(row['airline']), 
                    from_to_normalize(row["from"]), 
                    stop_normalize(row["stop"]), 
                    dep_arr_time_normalize(row["arr_time"]), 
                    from_to_normalize(row["to"]), 
                    float(row["class"])
                ]
            row = np.array(row).reshape(1, -1)
            result.append(model.predict(row))
        return jsonify({'prediction': [i.tolist()[0][0] for i in result]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)