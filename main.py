from transformers import AutoTokenizer
import os



os.environ["TOKENIZERS_PARALLELISM"] = "false"

ckpt = "bert-base-cased"
tokenzier = AutoTokenizer.from_pretrained(ckpt)

text = "my name is swayam"
encoding = tokenzier(text, return_offsets_mapping=True)

print(encoding)
start, end = encoding.word_to_chars(3)
print(text[start:end])

if __name__ == "__main__":
    pass
