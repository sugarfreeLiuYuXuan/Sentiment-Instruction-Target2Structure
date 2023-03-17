from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir="cache_t5")

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", cache_dir="cache_t5")