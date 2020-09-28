import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

#le = pickle.load(open("models/labelencoder.sav", 'rb'))
bow_transformer = pickle.load(open("models/bow_transformer.sav", 'rb'))
model = pickle.load(open("models/model.sav", 'rb'))

lemmatiser = WordNetLemmatizer()
def text_process(tex):
	# 1. Removal of Punctuation Marks 
	nopunct=[char for char in tex if char not in string.punctuation]
	nopunct=''.join(nopunct)
	# 2. Lemmatisation 
	a=''
	i=0
	for i in range(len(nopunct.split())):
		b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
		a=a+b+' '
	# 3. Removal of Stopwords
	return ' '.join([word for word in a.split() if word.lower() not 
			in stopwords.words('english')])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	df = pd.DataFrame()
	df['text'] = list(request.form.values())
	df['text'] = df.text.astype(str)
	df['text'] = df['text'].apply(text_process)
	X = df['text']
	X = bow_transformer.transform(X)
	prediction = model.predict(X)
	output = ['EAP', 'HPL', 'MWS']
	
	return render_template('index.html', prediction_text='Predicted Author is {}'.format(output[prediction[0]]))


if __name__ == "__main__":
    app.run(debug=True)