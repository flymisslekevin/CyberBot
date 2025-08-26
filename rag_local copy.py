import json, textwrap, numpy as np, faiss, ollama

INDEX_PATH   = "vectors/faiss.index"
MAPPING_PATH = "vectors/mapping.json"
EMB_MODEL    = "nomic-embed-text"
CHAT_MODEL   = "phi"
TOP_K        = 2
DEBUG        = False  # flip to False when you’re done

print("🔹 loading FAISS index …")
index   = faiss.read_index(INDEX_PATH)
dim     = index.d
mapping = json.load(open(MAPPING_PATH))

def embed(text: str) -> np.ndarray:
    vec = ollama.embeddings(model=EMB_MODEL, prompt=text)["embedding"]
    arr = np.asarray(vec, dtype="float32").reshape(1, dim)
    faiss.normalize_L2(arr)
    return arr

def ask(question: str):
    q_vec = embed(question)
    D, I = index.search(q_vec, TOP_K)  # D = distances, I = ids

    if DEBUG:
        print("\n🔍  FAISS raw results")
        for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
            meta = mapping[int(idx)]
            snippet = meta["text"].replace("\n", " ")
            print(f"{rank:2d}. id={idx:<6}  dist={dist:.4f}  {snippet}…")

    context_blocks = [mapping[int(i)]["text"] for i in I[0]]
    context = "\n\n".join(context_blocks)

    if DEBUG:
        token_estimate = len(context.split())
        print(f"\n🗃️  Assembled context: {len(context_blocks)} chunks, ~{token_estimate} tokens")

    prompt = textwrap.dedent(f"""
    You are an expert on U.S. health-insurance benefits.
    Answer the question using ONLY the context below.
    If the answer is not contained in the context, say “I don’t know.”.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """).strip()

    reply = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,        # change to True for token-streaming
    )
    return reply["message"]["content"], context_blocks

if __name__ == "__main__":
    while True:
        try:
            q = input("\n❓  Ask a question (or 'q' to quit): ")
            if q.lower().strip() in {"q", "quit", "exit"}:
                break
            ans, ctx = ask(q)
            print("\n🟢  Answer:\n", ans)
        except KeyboardInterrupt:
            break
