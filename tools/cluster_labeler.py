import numpy as np,os
from tools.watsonx_utils import wx_embeddings,inference_llm_dutch
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()  # load .env vars once

MAX_CHARS = int(os.getenv("LLM_INPUT_MAX_CHARS", 40000))

def process_text(first_page_text: str, filename: str = "unknown.pdf"):
    """
    Try LLM with shrinking logic for token-limit errors.
    Always return a result object.
    If any error occurs, return Unknown + actual error message.
    """
    if not first_page_text or len(first_page_text.strip()) < 100:
        return {
            "filename": filename,
            "document_label": "Unknown",
            "explanation": "Geen tekst beschikbaar voor analyse (first_page_text is None of leeg).",
            "status": "Error",
        }
    snippet = first_page_text[:MAX_CHARS]
    MIN_LIMIT = 8000

    while True:
        try:
            result = inference_llm_dutch(snippet)
            # success → break loop
            break

        except Exception as e:
            error_msg = str(e)

            # detect token limit problems
            msg = error_msg.lower()
            if "token" in msg or "input tokens" in msg or "exceed" in msg:
                new_len = int(len(snippet) * 0.5)

                if new_len < MIN_LIMIT:
                    # return the actual WatsonX error
                    return {
                        "filename": filename,
                        "document_label": "error",
                        "explanation": error_msg,
                        "status": "Error",
                    }

                print(f"⚠️ Token limit exceeded. Retrying with smaller size: {new_len}")
                snippet = snippet[:new_len]
                continue

            # non-token-limit error → return actual error
            return {
                "filename": filename,
                "document_label": "error",
                "explanation": error_msg,
                "status": "Error",
            }

    # successful LLM output
    return {
        "filename": filename,
        "document_label": result.get("label", "Unknown").strip(),
        "explanation": result.get("explanation", "").strip(),
        "status": "OK",
    }


def infer_cluster_label(cluster_df,  sample_size: int = None,similarity_threshold: float = None):
    """
    Infer a common label for a cluster of documents using majority voting + semantic similarity.
    Works for any sample size (3, 5, etc.)
    """
     # --- Step 0: Load defaults from environment if not passed ---
    sample_size = sample_size or int(os.getenv("CLUSTER_SAMPLE_SIZE", 3))
    similarity_threshold = similarity_threshold or float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    print(f" Using sample_size={sample_size}, similarity_threshold={similarity_threshold}")

    # --- Step 1: sample documents ---
    sample_rows = cluster_df.sample(n=min(sample_size,len(cluster_df)), random_state=42)
    label_records = []  # store {filename, label}
    labels_only = []

    for _, row in sample_rows.iterrows():
        print(f"Processing cluster {row['cluster_id']} | file: {row['filename']}")
        result = process_text(row["firstpagetxt"], row["filename"])
        label = result.get("document_label", "").strip()
        explanation = result.get("explanation", "").strip()
        label_records.append({"filename": row["filename"], "label": label,"explanation": explanation})
        labels_only.append(label)
        # if label:
        #     labels.append(label)
    if not labels_only:
        return {
            "cluster_label": "Unknown",
            "labels": [],
            "status": "Empty",
            "similarity_score": 0.0,
            "sample_files": []
        }

    # --- Step 2: Majority vote ---
    unique_labels, counts = np.unique(labels_only, return_counts=True)
    top_label = unique_labels[np.argmax(counts)]
    majority_ratio = counts.max() / len(labels_only)

    # If majority (e.g. 2/3 or 3/5), mark as Auto
    if majority_ratio >= 0.6:  # two-thirds threshold
        return {
            "cluster_label": top_label,
            "labels": label_records,
            "status": "Auto",
            "similarity_score": 1.0
        }

    # --- Step 3: Semantic similarity (Watsonx embeddings) ---
    try:
        emb_results = wx_embeddings.embed_documents(texts=labels_only)
        embeddings = [e if isinstance(e, list) else e.get("embedding", []) for e in emb_results]

        sim_matrix = cosine_similarity(embeddings)
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        avg_similarity = float(np.mean(upper_triangle))

    except Exception as e:
        print("⚠️ Embedding similarity failed:", e)
        avg_similarity = 0.0

    # --- Step 4: High similarity → Auto-Similar ---
    if avg_similarity >= similarity_threshold:
        return {
            "cluster_label": top_label,
            "labels": label_records,
            "status": "Auto-Similar",
            "similarity_score": round(avg_similarity, 3)
        }

    # --- Step 5: Fallback → Manual Review ---
    return {
        "cluster_label": "ManualReview",
        "labels": label_records,
        "status": "Manual",
        "similarity_score": round(avg_similarity, 3)
    }


