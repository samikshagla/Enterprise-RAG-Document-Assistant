import streamlit as st
import requests

# Constants for the API backend (running on localhost)
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Enterprise RAG Document Assistant")
st.markdown("Upload a PDF to ingest knowledge, then ask questions about it below!")

# -----------------
# SIDEBAR: File Upload
# -----------------
with st.sidebar:
    st.header("1. Upload Knowledge Base")
    st.write("Upload a PDF document to update the Vector Database context.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Uploading and indexing... this may take a moment."):
                try:
                    # Construct a multipart request file payload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"Success! Embedded: {uploaded_file.name}")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown backend error.')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}. Is the FastAPI server running?")
        else:
            st.warning("Please select a PDF file first.")
            
    st.divider()
    st.markdown("*Note: Requires the backend server to be running on localhost:8000.*")

# -----------------
# MAIN AREA: Chat UI
# -----------------
st.header("2. Chat Interface")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user chat input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("▌ (Thinking...)")
        
        try:
            # Query the FastAPI backend
            payload = {"query": prompt}
            res = requests.post(f"{API_BASE_URL}/chat", json=payload)
            res.raise_for_status()
            
            answer = res.json().get("answer", "No answer returned.")
            
            # Display final text
            message_placeholder.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to backend server. Make sure `uvicorn backend:app` is running!"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": f"**ERROR:** {error_msg}"})
        except Exception as e:
            error_msg = f"Failed to get a response: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": f"**ERROR:** {error_msg}"})
