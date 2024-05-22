from transformers import BertTokenizer, BertForQuestionAnswering
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', ignore_mismatched_sizes=True).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, context):
    # Encode the inputs, ensuring tensors are sent to the same device as model
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs.input_ids[0, answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens)
    return answer

context = "Hugging Face is a company that specializes in creating natural language processing software."

while True:
    question = input("\nAsk a question (type 'q' to quit): \n")
    if question.lower() == 'q':
        print("Exiting...")
        break
    answer = answer_question(question, context)
    print("Answer:", answer)
