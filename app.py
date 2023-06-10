from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.piplines.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


@app.route('/',methods=['GET','POST'])
def predictor():
    if request.method == 'POST':
        return render_template('index.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('writing_score'),
            writing_score=request.form.get('reading_score')
        )

        pred_df = data.get_data_as_data_frame()

        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)