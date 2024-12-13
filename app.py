
import os
import tempfile
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain

class EnhancedPDFChatbot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Retriever will be set when documents are loaded
        self.retriever = None

        # Create Chat History Store
        self.store = {}

    def load_pdf_documents(self, uploaded_files):
        # Collect all documents from uploaded PDFs
        all_splits = []
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Process each uploaded PDF
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                
                # Split documents
                splits = text_splitter.split_documents(docs)
                all_splits.extend(splits)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                # Remove temporary file
                os.unlink(temp_file_path)
        
        # Create and set retriever from PDF documents
        if all_splits:
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.embeddings)
            self.retriever = vectorstore.as_retriever()
            return True
        return False

    def _create_conversational_rag_chain(self):
        # Check if retriever is set
        if self.retriever is None:
            raise ValueError("No documents have been loaded. Please upload PDF files.")

        # System Prompt
        system_prompt = (
            "You are a helpful assistant for answering questions based on the uploaded PDF documents. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer from the context, "
            "say that you don't know. Keep your answers concise and directly "
            "related to the uploaded documents.\n\n"
            "{context}"
        )

        # Contextualize Question Prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Create History Aware Retriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # QA Prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Create Chains
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Create Session History Function
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        # Wrap RAG Chain with Message History
        return RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def chat(self, input_text, session_id):
        try:
            # Ensure conversational RAG chain is created with current retriever
            conversational_rag_chain = self._create_conversational_rag_chain()
            
            response = conversational_rag_chain.invoke(
                {"input": input_text},
                config={"configurable": {"session_id": session_id}}
            )
            return response['answer']
        except ValueError as ve:
            return str(ve)
        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    # Set page configuration
    st.set_page_config(
        page_title="PDF Q&A Chatbot",
        page_icon="📄",
        layout="wide"
    )

    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stChatMessage {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🤖 DocuBot")
    st.markdown("Upload PDFs and ask questions about their content!")

    # Initialize the chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = EnhancedPDFChatbot()
    
    # PDF Upload Section
    st.sidebar.header("📤 Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or more PDF documents to start chatting"
    )
    
    # Documents processing
    if uploaded_files:
        with st.sidebar:
            if st.button("🔍 Process Documents", key="process_docs"):
                with st.spinner("Processing PDFs..."):
                    success = st.session_state.chatbot.load_pdf_documents(uploaded_files)
                    if success:
                        st.success("📊 Documents processed successfully!")
                        st.session_state.docs_processed = True
                    else:
                        st.error("❌ No valid documents were processed.")
                        st.session_state.docs_processed = False
    
    # Main chat interface
    if not uploaded_files:
        # No documents uploaded state
        st.warning("📋 Please upload PDF documents in the sidebar to start chatting.")
        st.info("Steps to use the chatbot:\n1. Upload PDF files in the sidebar\n2. Click 'Process Documents'\n3. Start asking questions about the uploaded documents")
    elif not hasattr(st.session_state, 'docs_processed') or not st.session_state.docs_processed:
        # Documents not processed
        st.warning("📚 Documents are uploaded but not processed. Click 'Process Documents' in the sidebar.")
    else:
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Display user message
            st.chat_message("user").markdown(prompt)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get chatbot response
            response = st.session_state.chatbot.chat(
                input_text=prompt, 
                session_id=st.session_state.get('session_id', 'default_session')
            )

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()