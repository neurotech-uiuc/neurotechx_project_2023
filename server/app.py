from flask import Flask, request, jsonify
import csv


app = Flask(__name__)

@app.route('/post_direction', methods=['POST'])
def post_number():
    data = request.get_json()

    if 'direction' not in data:
        return jsonify({"error": "Missing 'direction' in request data"}), 400

    
    dir = str(data['direction'])
    if not isinstance(dir, str):
        return jsonify({"error": "Invalid 'direction' value"}), 400
    
    # Append the received data to a CSV file
    with open('dirs.csv', 'a', newline='') as csvfile:
        fieldnames = ['direction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Uncomment the following line if you want to write the header for the first time
        # writer.writeheader()
        
        writer.writerow({'direction': dir})

    return jsonify({"result": dir})

if __name__ == '__main__':
    app.run(debug=True)