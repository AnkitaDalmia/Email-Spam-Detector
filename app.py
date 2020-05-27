from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

new_word_dict = pickle.load(open("mystrings.pkl", "rb"))
clf = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['post'])
def predict():
    email = request.form.get('yeemailhai')

    input_email = []

    for i in new_word_dict:
        input_email.append(email.count(i[0]))

    input_email = np.array(input_email)
    result = clf.predict(input_email.reshape(1, 3000))[0]

    if result == 1:
        return render_template("index.html", result=1)
    else:
        return render_template("index.html", result=-1)


if __name__ == "__main__":
    app.run(debug=True)
