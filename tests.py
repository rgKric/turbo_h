from transformers import AutoTokenizer, T5ForConditionalGeneration
from functools import partial

saved_checkpoint = './t5_model'
tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(saved_checkpoint)

print(model)