from transformers import BertTokenizer, BertForQuestionAnswering
import torch

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Find the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    # Convert indices to tokens, then tokens to string
    answer_tokens = inputs.input_ids[0, answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens)
    
    return answer
    
if __name__ == '__main__':
    question = "What is the name of the repository?"
    context = "The name of the repository is 'transformers'."
    answer = answer_question(question, context)
    print(answer)  # Output: transformers