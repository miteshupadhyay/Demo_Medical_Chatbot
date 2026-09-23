# Demo_Medical_Chatbot

# with openai and pinecone as vector DB

# Create Virtual Env

uv init
uv venv
.venv\Scripts\activate

uv add -r requirements.txt

# Create Folder with template file
bash template.sh

# Run Store index to create and store embeddings into Vector DB
python store_index.py

# Run the application
python app.py

# Access the application
http://localhost:8080