from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Initialize the summarization models
abstractive_summarizer = pipeline('summarization', model='t5-small')
extractive_summarizer = pipeline('summarization', model='google/pegasus-xsum')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Summarization result page
@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        abstractive_summary = abstractive_summarizer(text, max_length=150, min_length=30, do_sample=False)
        extractive_summary = extractive_summarizer(text, max_length=150, min_length=30, do_sample=False)
        return render_template('result.html', text=text, abstractive_summary=abstractive_summary[0]['summary_text'], extractive_summary=extractive_summary[0]['summary_text'])

if __name__ == '__main__':
    app.run(debug=True)
