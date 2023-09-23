from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def generate_text(tokenizer, model,tokens_len,temperature=.7):
    prompt = "<|startoftext|>"
    inputs = tokenizer(prompt, return_tensors="pt")
    sample_output = model.generate(**inputs,
                                    do_sample = True, 
                                    min_new_tokens = tokens_len,                           
                                    max_new_tokens  = tokens_len+3, 
                                    temperature = temperature,
                                    top_k = 50, 
                                    top_p = 0.92, 
                                    num_return_sequences = 1,
                                    pad_token_id=tokenizer.eos_token_id)

    text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return text



def main():
    parser = argparse.ArgumentParser(description='Generate Text')
    parser.add_argument('--length', type=int, default=50, metavar='L',
                        help='Generated text length (tokens)')
    parser.add_argument('--temperature', type=float, default=0.7, metavar='T',
                        help='Diversity of output text')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("results",local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("results",local_files_only=True)
    
    text = generate_text(tokenizer,model,args.length,args.temperature)
    print("Output:\n" + 100 * '-')
    print(text)

    
if __name__=='__main__':
    main()

