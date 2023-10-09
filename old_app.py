import os
import re
import tempfile
import streamlit as st
from streamlit_chat import message
from src.agent import Agent
from src.utils import sanitize_filename
from dotenv import load_dotenv




# Set header image
st.image("https://www.writepal.ai/images/writepal-web1.png", width=300)

# set title
st.title("ChatPal🖋")
# st.set_page_config(page_title="ChatPal")
    
def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["agent"].ask(user_text)
        print(agent_text)
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    st.session_state["agent"].forget()  # to reset the knowledge base
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        # Check if the file object has the 'name' attribute
        if not hasattr(file, 'name'):
            continue

        # Sanitize file name
        sanitized_name = sanitize_filename(file.name)
        
        if sanitized_name in st.session_state["files_set"]:
            continue
        
        st.session_state["files_set"].add(sanitized_name)

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, sanitized_name)
            
            # Write the uploaded file data to the file_path
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())

            # Process the file
            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {sanitized_name}"):
                st.session_state["agent"].ingest(file_path)

                
                

def is_openai_api_key_set() -> bool:
    return len(st.session_state["OPENAI_API_KEY"]) > 0

def main():
    load_dotenv()
    if len(st.session_state) == 0:
        st.session_state["files_set"] = set()
        st.session_state["OPENAI_API_KEY"] = ""
        st.session_state["messages"] = []
        if is_openai_api_key_set():
            st.session_state["agent"] = Agent()
        else:
            st.session_state["agent"] = None

    if st.text_input("OpenAI API Key", value=st.session_state["OPENAI_API_KEY"], key="input_OPENAI_API_KEY", type="password"):
        if (
            len(st.session_state["input_OPENAI_API_KEY"]) > 0
            and st.session_state["input_OPENAI_API_KEY"] != st.session_state["OPENAI_API_KEY"]
        ):
            st.session_state["OPENAI_API_KEY"] = st.session_state["input_OPENAI_API_KEY"]
            if st.session_state["agent"] is not None:
                st.warning("Please, upload the files again.")
            st.session_state["messages"] = []
            st.session_state["user_input"] = ""
            st.session_state["agent"] = Agent(st.session_state["OPENAI_API_KEY"])

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        disabled=not is_openai_api_key_set(),
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", disabled=not is_openai_api_key_set(), on_change=process_input)
    st.divider()
    


if __name__ == "__main__":
    main()
