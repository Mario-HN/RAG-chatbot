import openai
import streamlit as st
from assistant import ConversationalChain, initialize_state, load_pdf, load_qa_chain
from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(find_dotenv()) 

def main():
    i = 0
    # Set up the Streamlit page configuration and title
    st.set_page_config(
        page_title="TSA Assistant",
        layout="wide",
    )

    # file uploads and OpenAI keys

    file_path = "/home/mario/Downloads/RD.pdf"
    _, file_pages = load_pdf(file_path=file_path)
    openai_keys = os.getenv("OPENAI_API_KEY")

    st.markdown("***")
    st.subheader("Interação com usuário.")

    # Initialize the session state variables
    initialize_state()

    # Add a flag in the session state for API key validation
    if "is_api_key_valid" not in st.session_state:
        st.session_state.is_api_key_valid = None
    # Load the QA chain if documents and OpenAI keys are provided, and handle OpenAI AuthenticationError
    CC = load_qa_chain(file_pages, openai_keys)
    if file_pages and openai_keys and not st.session_state.qa_chain:
        try:
            st.session_state.qa_chain = CC.create_qa_chain()
            st.session_state.is_api_key_valid = True  # Valid API key
        except openai.AuthenticationError as e:
            st.error(
                'Forneça uma chave de API válida. Atualize a chave de API na barra lateral e clique em "Concluir configuração" para prosseguir.',
                icon="🚨",
            )
            st.session_state.is_api_key_valid = False  # Invalid API key

    # Enable the chat section if the QA chain is loaded and API key is valid
    if st.session_state.qa_chain and st.session_state.is_api_key_valid:
        st.success("Configuração concluída")
        prompt = st.chat_input(
            "Faça perguntas sobre os documentos enviados", key="chat_input"
        )

        # Process user prompts and generate responses
        if prompt and (
                st.session_state.messages[-1]["content"] != prompt
                or st.session_state.messages[-1]["role"] != "user"
        ):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner(
                    "Recuperando informações relevantes e gerando resultados..."
            ):
                response = st.session_state.qa_chain.run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        # Display the conversation messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                # if message["role"] == "assistant":
                #     st.write("Texto recuperado:")
                #     st.write(prompt)
        # except:
        #       st.write("")
        #     st.write("Texto recuperado:")
        #     st.write("Não foi possível recuperar o chunk...")

    else:
        st.info("Carregando...")
        # Disable the chat input if the API key is invalid
        no_prompt = st.chat_input(
            "Faça perguntas sobre os documentos enviados",
            disabled=not st.session_state.is_api_key_valid,
        )



if __name__ == "__main__":
    main()
