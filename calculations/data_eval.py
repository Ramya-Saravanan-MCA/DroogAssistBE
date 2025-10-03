import json
from typing import List, Dict
import pandas as pd
from retrieval.retriever import Retriever
from sentence_transformers import SentenceTransformer, util

with open("goal_set.json", "r") as file:
    goal_set = json.load(file)

def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return len(set(retrieved_k) & set(relevant)) / k

def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    return len(set(retrieved_k) & set(relevant)) / len(relevant)

def reciprocal_rank(retrieved: List[int], relevant: List[int], k: int) -> float:
    for idx, chunk_id in enumerate(retrieved[:k]):
        if chunk_id in relevant:
            return 1.0 / (idx + 1)
    return 0.0

def evaluate_retrieval(
    goal_set: List[Dict], 
    retriever, 
    k: int = 10, 
    retrieval_mode="hybrid", 
    k_dense=10, 
    k_sparse=10, 
    rrf_k=60, 
    top_k_final=10,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    rows = []
    embedder = SentenceTransformer(embedding_model_name)
    answer_similarities = []

    for q in goal_set:
        queries = [q['query']] + q.get('paraphrased_queries', [])
        relevant = q['mapped_chunk_ids']

        # Ensure golden and generated answer exist for each query
        golden_answer = q.get('golden_answer', "")
        generated_answer = q.get('generated_answer', "")

        for user_q in queries:
            results, _ = retriever.retrieve(
                user_q,
                mode=retrieval_mode,
                k_dense=k_dense,
                k_sparse=k_sparse,
                rrf_k=rrf_k,
                top_k_final=top_k_final,
                doc_filter=None
            )
            retrieved = list(results['chunk_id']) if not results.empty else []

            p_at_k = precision_at_k(retrieved, relevant, k)
            r_at_k = recall_at_k(retrieved, relevant, k)
            rr = reciprocal_rank(retrieved, relevant, k)

            # Compute answer-level semantic similarity
            if golden_answer.strip() and generated_answer.strip():
                emb_gold = embedder.encode(golden_answer, convert_to_tensor=True)
                emb_gen = embedder.encode(generated_answer, convert_to_tensor=True)
                semantic_sim = float(util.cos_sim(emb_gen, emb_gold)[0][0])
                answer_similarities.append(semantic_sim)
            else:
                semantic_sim = float('nan')

            rows.append({
                "query_id": q['query_id'],
                "user_query": user_q,
                "relevant_chunk_ids": relevant,
                "retrieved_chunk_ids": retrieved[:k],
                "Precision@k": p_at_k,
                "Recall@k": r_at_k,
                "ReciprocalRank@k": rr,
                "GoldenAnswer": golden_answer,
                "GeneratedAnswer": generated_answer,
                "AnswerSemanticSimilarity": semantic_sim
            })

    df = pd.DataFrame(rows)
    print(df[["query_id", "user_query", "Precision@k", "Recall@k", "ReciprocalRank@k", "AnswerSemanticSimilarity"]])

    precision_k = df["Precision@k"].mean()*100
    recall_k = df["Recall@k"].mean()*100
    mrr_k = df["ReciprocalRank@k"].mean()*100
    mean_answer_sim = pd.to_numeric(df["AnswerSemanticSimilarity"], errors='coerce').mean()

    print(f"\nOverall Results (k={k}):")
    print(f"Precision@{k}: {precision_k:.3f}")
    print(f"Recall@{k}: {recall_k:.3f}")
    print(f"MRR@{k}: {mrr_k:.3f}")
    print(f"Mean Answer Semantic Similarity: {mean_answer_sim:.3f}")

    return df

if __name__ == "__main__":
    retriever = Retriever(db_path="rag_chatbot/data/lancedb", table_name="unified_knowledge_base")
    K = 10
    evaluate_retrieval(goal_set, retriever, k=K)