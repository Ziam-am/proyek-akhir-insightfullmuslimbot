# =[Modules dan Packages]========================
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from flask_ngrok import run_with_ngrok

import pandas as pd
import re
import pickle
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# download nltk
preparation()

# =[Variabel Global]=============================
app = Flask(__name__, static_url_path='/static')
model = None

# Load model LSTM
def load_lstm_model():
    global model
    model = load_model('chatbot_model.h5')
    # Optional: Lakukan persiapan lain yang diperlukan seperti pengaturan tokenizer, dll.

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk Halaman aplikasi]
@app.route("/aplikasi")
def chatbot():
    return render_template('aplikasi.html')

# [Routing untuk API]
@app.route("/api/deteksi", methods=['POST'])
def api_deteksi():
    if request.method == 'POST':
        # Ambil input dari pengguna
        input_text = request.form['input_text']
        
        # Lakukan preprocessing pada input_text jika diperlukan
        prediksi_input = text_preprocessing_process(prediksi_input)
        
        # Lakukan prediksi menggunakan model LSTM
        prediksi = lstm_predict(input_text)
        
        if prediksi == 0:
            hasil_prediksi = "berhasil"
        else:
            hasil_prediksi = "pertanyaan tidak dimengerti"
            
        # Return hasil prediksi dengan format JSON
        return jsonify({
            "data": hasil_prediksi,
        })
        #return jsonify({"prediksi": prediksi})

# Prediksi menggunakan model LSTM
def lstm_predict(input_text):
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result
    
    return prediksi

# Main
if __name__ == '__main__':


    # Run Flask di localhost 
      run_with_ngrok(app)
      app.run()
