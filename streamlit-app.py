import streamlit as st
import time
from transformers import TranslationPipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    translation_pipe = TranslationPipeline(model=model, tokenizer=tokenizer, task='translation')
    return translation_pipe

def stream_data(answer):
    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.02)

def main():
    st.title("Translator")
    translator = load_model()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer = translator(prompt)[0]['translation_text']
            response = st.write_stream(stream_data(answer))
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
