from flask import Flask, render_template, url_for, request

#intialize a Flask object 'app', name = name of the current module
app = Flask(__name__)

'''all methods and processing done here'''
'''this method will be called by default then we run the flask web app'''
@app.route('/')
def home():
    return render_template('home.html')

'''this method will be called by default then we run the flask web app'''
@app.route('/predict',methods=['POST'])
def predict():
    if (request.method == 'POST'):
        comment = request.form['comment']
        data = [comment]
        
        '''transforming the data into countvectorizer'''
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        
        final_comment = cv.transform(data).toarray()
        
        '''loading the model'''
        from sklearn.externals import joblib
        model = joblib.load('saved_model.pkl')
        
        '''predicting the comment'''
        model.predict(final_comment)
    return render_template('predict.html')
    

'''execution begins here'''
if __name__ == '__main__':
    app.run(debug=True)

