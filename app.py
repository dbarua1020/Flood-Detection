from flask import Flask, request, render_template
import joblib

app = Flask(__name__) 
model = joblib.load('newmodel.joblib') 

@app.route('/', methods=['GET'])
def home():
    return render_template('services.html')

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        date = int(request.form['date'])
        precipitation = float(request.form['precipitation'])
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        wind = float(request.form['wind'])
        prediction = model.predict([[date, precipitation, temp_max, temp_min, wind]])
        y_pred = prediction[0]
        if y_pred==0:
            output="Drizzle"
        elif y_pred==1:
            output="Fog"
        elif y_pred==2:
            output="Rain"
        elif y_pred==3:
            output="Snow"
        elif y_pred==4:
            output="Sun"
        return render_template('services.html', output="The forecast of Weather is {}".format(output))
if __name__ == '__main__':
   app.run(debug=True)