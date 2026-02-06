import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load embedding model (must match what you used during ingestion)
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load Chroma vector store
chroma = Chroma(
    persist_directory="chroma_db",  # folder uploaded to Hugging Face Space
    embedding_function=embedding
)
retriever = chroma.as_retriever()

# 3. Load Mistral 7B (or adjust if you want a smaller model like Phi-2 or Gemma-2B)
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=200
)
llm = HuggingFacePipeline(pipeline=pipe)

# 4. Define RetrievalQA chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5. Define the chatbot logic
def ask_big_beautiful_bill(question):
    if not question.strip():
        return "Please enter a question.", ""

    result = qa_chain(question)
    answer = result["result"]
    source_docs = result["source_documents"]
    
    # Show context from top 3 docs with metadata
    context = "\n\n".join(
        f"[Page: {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('source', '')}]\n{doc.page_content[:300]}..."
        for doc in source_docs[:3]
    )

    return answer, context

# 6. Launch the Gradio interface
demo = gr.Interface(
    fn=ask_big_beautiful_bill,
    inputs=gr.Textbox(label="Ask a question about the Big Beautiful Bill:"),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Retrieved Context (from bill)")
    ],
    title="📜 Big Beautiful Bill Chatbot",
    description="Ask questions about the 1,000-page U.S. policy bill. Answers are retrieved and generated from real text."
)

demo.launch()
