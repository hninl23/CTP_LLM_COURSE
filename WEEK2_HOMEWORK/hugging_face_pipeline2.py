from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
import torch.nn.functional as F


model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer=  AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
X_train = ["I have been waiting for HuggingFace course my whole life.",
            "Python is great!"]

result = classifier(X_train)
print("Result:", result)

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print("Batch:", batch)

# #
# Batch: {'input_ids': tensor([[  101,  1045,  2031,  2042,  3403,  2005, 17662, 12172,  2607,  2026,
#           2878,  2166,  1012,   102],
#         [  101, 18750,  2003,  2307,   999,   102,  0,     0,     0,     0, 0,     0,     0,     0]]), 
#         'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
###

#MAKING INFERENCE?
with torch.no_grad():
    outputs = model(**batch) # ** = unpatch the dictioanry "batch"
    print("Outputs:" , outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print("Predictions:", predictions)
    labels = torch.argmax(predictions, dim=1)
    print("Labels:", labels)

###
# Outputs: SequenceClassifierOutput(loss=None, 
#         logits=tensor([[-2.3642,  2.4149],
#         [-4.2745,  4.6111]]), 
#         hidden_states=None, attentions=None)
# Predictions: tensor([[8.3338e-03, 9.9167e-01],
#         [1.3835e-04, 9.9986e-01]])
# Labels: tensor([1, 1])
###

#SAVE/LOAD
save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tok = AutoTokenizer.from_pretrained(save_directory)
mod = AutoModelForSequenceClassification.from_pretrained(save_directory)