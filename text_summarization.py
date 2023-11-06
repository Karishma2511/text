# Install the transformers library
import tensorflow as tf
from transformers import pipeline

# Define a function for abstractive summarization
def abstractive_summarization(text):
    summarizer = pipeline('summarization', model='t5-small', revision='main',max_length=55)
    #summarizer = pipeline('summarization')
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Define a function for extractive summarization
def extractive_summarization(text):
    summarizer = pipeline('summarization', model='google/pegasus-xsum',max_length=55)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Example usage
text = """
    Text summarization is the process of creating a concise and coherent summary of a longer text while preserving its key information. There are two main approaches to text summarization: abstractive and extractive. Abstractive summarization generates summaries by understanding the text and paraphrasing it, while extractive summarization selects and combines existing sentences from the text. Both approaches have their advantages and trade-offs, and the choice between them depends on the specific use case and requirements.
"""

abstractive_summary = abstractive_summarization(text)
print("Abstractive Summary:")
print(abstractive_summary)

extractive_summary = extractive_summarization(text)
print("\nExtractive Summary:")
print(extractive_summary)
