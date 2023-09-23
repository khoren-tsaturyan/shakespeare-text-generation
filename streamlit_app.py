import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer


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


st.title('Shakespeare Text Generation')

col1, col2 = st.columns(2)
with col1:
    generated_lengh = st.number_input('Number of tokens to be generated', 2, 400, 50)
with col2:
    temperature = st.number_input('Temperature', 0., 100., 0.7,step=0.1)

go = st.button('Generate')
if go:
    try:
        tokenizer = AutoTokenizer.from_pretrained("results",local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("results",local_files_only=True)
    
        generate_text = generate_text(tokenizer,model,generated_lengh,temperature)
        c = st.container()
        for text in generate_text.split('\n'):
            c.write(text)
        print(generate_text)
        
    except Exception as e:
        st.exception("Exception: %s\n" % e)

