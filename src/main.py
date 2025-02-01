from transformers import AutoTokenizer, pipeline, GPT2LMHeadModel
from transformers import AutoConfig
 
model_name = "/home/nil/python_projects/gpt2_finetuned_45k_10epochs/results/model"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# model.eval()
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
config = AutoConfig.from_pretrained(model_name)
 
# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)
 
while True:
    user_input = input("You: ") 
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
 
    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
 
    attention_mask = inputs['attention_mask']
 
    response = text_generator(
        user_input,
        max_length=200,  
        num_return_sequences=1,
        do_sample=True,
        temperature=0.1,
        truncation=False,
    )
 
    suggestion = response[0]['generated_text']
    suggestion = suggestion[len(user_input):].strip()  
    print(f"Bot: {suggestion}")