from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel



classifier = pipeline("sentiment-analysis")
result = classifier("I have been waiting for HuggingFace course my whole life.")

print(result)

#EXPERIMENT WITH TOKENS AND MODELS
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer=  AutoTokenizer.from_pretrained(model_name)

sequence = "Using a tranformer network is simple"
token_result = tokenizer(sequence)
print("token_result:", token_result)
tokens = tokenizer.tokenize(sequence)
print("tokens", tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print("ids:", ids)
decoded_string = tokenizer.decode(ids)
print("decoded_string:", decoded_string)

#TRYING OUT DIFFERENT PIPELINES
generator = pipeline("text-generation", model="distilgpt2")

text_result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(text_result)

zero_shot_classifier = pipeline("zero-shot-classification")

classifier_result = zero_shot_classifier(
    "This is a couse about Python list comprehension",
    candidate_labels=["education", "politics", "business"]
)

print(classifier_result)

#SUMMARIZATION PIPELINE
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = """Hugging Face, Inc. is a French-American company based in New York City that develops computer tools for building applications using machine learning. It is most notable for its transformers library built for natural language processing applications and its platform that allows users to share machine learning models and datasets and showcase their work.
            The company was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf in New York City, originally as a company that developed a chatbot app targeted at teenagers.[1] The company was named after the "hugging face" emoji.After open sourcing the model behind the chatbot, the company pivoted to focus on being a platform for machine learning. In March 2021, Hugging Face raised US$40 million in a Series B funding round.
            On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model.[3] In 2022, the workshop concluded with the announcement of BLOOM, a multilingual large language model with 176 billion parameters.[4][5]"""

print("Text_Summary:", summarizer(text, max_length=100, min_length= 30, do_sample=False))