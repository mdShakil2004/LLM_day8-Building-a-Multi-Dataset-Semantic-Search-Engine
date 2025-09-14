#  Semantic Search with Multiple Datasets (20k+ Docs)
# Datasets: 20 Newsgroups, AG News, Amazon Polarity, Wikipedia

!pip install sentence-transformers faiss-cpu scikit-learn ipywidgets datasets

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
import faiss
import numpy as np
import ipywidgets as widgets
from IPython.display import display, Markdown

# -----------------------------
#  Load Embedding Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
#  Dataset Loader
# -----------------------------
def load_dataset_choice(choice):
    if choice == "20 Newsgroups":
        train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
        test  = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'))
        documents = train.data + test.data
        targets   = list(train.target) + list(test.target)
        categories = train.target_names
    elif choice == "AG News":
        dataset = load_dataset("ag_news", split="train[:5000]")  # subset for speed
        documents = dataset["text"]
        targets = dataset["label"]
        categories = dataset.features["label"].names
    elif choice == "Amazon Polarity":
        dataset = load_dataset("amazon_polarity", split="train[:5000]")
        documents = dataset["content"]
        targets = dataset["label"]
        categories = dataset.features["label"].names
    elif choice == "Wikipedia":
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:5000]")  # subset
        documents = dataset["text"]
        targets = [0] * len(documents)  # no labels
        categories = None
    else:
        raise ValueError("Invalid dataset choice")
    
    return documents, targets, categories

# -----------------------------
# ⚡ Initialize with Default Dataset
# -----------------------------
dataset_choice = "20 Newsgroups"
documents, targets, categories = load_dataset_choice(dataset_choice)

print(f"Loaded {len(documents)} documents from {dataset_choice}")

# -----------------------------
# 🔢 Encode + Build FAISS Index
# -----------------------------
def build_index(docs):
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=64)
    dimension = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dimension)
    idx.add(np.array(embeddings))
    return idx, embeddings

index, doc_embeddings = build_index(documents)

# -----------------------------
# 🔍 Semantic Search Function
# -----------------------------
def semantic_search(query, top_k=5):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    
    results = []
    for j, i in enumerate(indices[0]):
        score = 1 / (1 + distances[0][j])  # normalize similarity
        snippet = documents[i].replace("\n", " ")[:300] + "..."
        category = categories[targets[i]] if categories else "Wikipedia"
        results.append((snippet, category, score))
    return results

# -----------------------------
# 🎛️ Widgets: Dataset Switcher + Search Box
# -----------------------------
dataset_dropdown = widgets.Dropdown(
    options=["20 Newsgroups", "AG News", "Amazon Polarity", "Wikipedia"],
    value="20 Newsgroups",
    description="Dataset:"
)

query_box = widgets.Text(
    description="Query:",
    placeholder="Type your search here..."
)

output = widgets.Output()

def on_dataset_change(change):
    global documents, targets, categories, index, doc_embeddings
    choice = change["new"]
    print(f"Switching to {choice}...")
    documents, targets, categories = load_dataset_choice(choice)
    index, doc_embeddings = build_index(documents)

def on_query_submit(change):
    with output:
        output.clear_output()
        query = change['new']
        if not query.strip():
            return
        results = semantic_search(query, top_k=5)
        
        display(Markdown(f"### 🔎 Query: **{query}**\n"))
        for snippet, category, score in results:
            display(Markdown(f"- **Category:** `{category}`  \n"
                             f"**Score:** {score:.4f}  \n"
                             f"**Snippet:** {snippet}\n"))

dataset_dropdown.observe(on_dataset_change, names='value')
query_box.observe(on_query_submit, names='value')

display(dataset_dropdown, query_box, output)
