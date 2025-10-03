import lancedb
import pandas as pd
import pyarrow as pa
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
class SessionLogger:
    def __init__(self, db_path: str):
        os.environ['AWS_REGION'] = os.getenv('AWS_REGION', 'ap-south-1')
        self.db = lancedb.connect(db_path)
        self._init_tables()
    def _get_timestamp(self):
        """Get current timestamp in Asia/Kolkata timezone"""
        return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    
    def _init_tables(self):
        buffer_schema = pa.schema([
            ("session_id", pa.string()),
            ("turn_id", pa.int64()),
            ("user_query", pa.string()),
            ("bot_response", pa.string()),
            ("timestamp", pa.string())
        ])
        
        items_schema = pa.schema([
            ("session_id", pa.string()),
            ("turn_id", pa.int64()),
            ("user_query", pa.string()),
            ("bot_response", pa.string()),
            ("summary", pa.string()),
            #details about models and retrieval
            ("llm_model", pa.string()),
            ("top_k_dense", pa.int64()),
            ("top_k_sparse", pa.int64()),
            ("top_k_combined", pa.int64()),
            ("intent_labels", pa.string()),

            ("intent_confidence", pa.float64()),
            ("slots", pa.string()),
            ("retrieval_type", pa.string()),
            ("retrieval_strength", pa.float64()),
            ("retrieved_chunk_ids", pa.string()),
            ("used_rag", pa.bool_()),
            ("answer_decision", pa.string()),
            ("embedding_latency_ms", pa.int64()),
            ("dense_retrieval_latency_ms", pa.int64()),
            ("sparse_retrieval_latency_ms", pa.int64()),
            ("llm_latency_ms", pa.int64()),
            ("total_latency_ms", pa.int64()),
            ("timestamp", pa.string()),
            # RAGAS evaluation fields - initially empty
            ("golden_answer", pa.string()),
            ("context_recall", pa.float64()),
            ("context_precision", pa.float64()),
            ("answer_correctness", pa.float64()),
            ("faithfulness", pa.float64()),
            ("answer_similarity", pa.float64()),
            # LLM cost/token stats
            ("router_input_tokens", pa.int64()),
            ("router_output_tokens", pa.int64()),
            ("router_total_tokens", pa.int64()),
            
            
            ("summarizer_input_tokens", pa.int64()),
            ("summarizer_output_tokens", pa.int64()),
            ("summarizer_total_tokens", pa.int64()),

            ("main_input_tokens", pa.int64()),
            ("main_output_tokens", pa.int64()),
            ("main_total_tokens", pa.int64()),

            ("router_cost", pa.float64()),
            ("summarizer_cost", pa.float64()),
            ("main_llm_cost", pa.float64()),
            
            ("total_turn_cost", pa.float64()),
        ])
        
        sessions_schema = pa.schema([
            ("session_id", pa.string()),
            ("total_turns", pa.int64()),
            ("llm_model", pa.string()),
            ("top_k_dense", pa.int64()),
            ("top_k_sparse", pa.int64()),
            ("top_k_combined", pa.int64()),
            ("retrieval_type", pa.string()),
            ("total_session_time_ms", pa.int64()),
            ("avg_embedding_latency_ms", pa.float64()),
            ("avg_dense_retrieval_latency_ms", pa.float64()),
            ("avg_sparse_retrieval_latency_ms", pa.float64()),
            ("avg_llm_latency_ms", pa.float64()),
            ("avg_retrieval_strength", pa.float64()),
            ("created_at", pa.string()),
            ("ended_at", pa.string()),

            #token details
            ("total_input_tokens", pa.int64()),
            ("total_output_tokens", pa.int64()),
            ("total_tokens", pa.int64()),
            ("total_cost_usd", pa.float64()),
        ])
    
        if "buffer" not in self.db.table_names():
            self.db.create_table("buffer", schema=buffer_schema)
        if "items" not in self.db.table_names():
            self.db.create_table("items", schema=items_schema)
        if "sessions" not in self.db.table_names():
            self.db.create_table("sessions", schema=sessions_schema)

    def log_to_buffer(self, session_id, turn_id, user_query, bot_response):
        buffer_table = self.db.open_table("buffer")
        df = buffer_table.to_pandas()
        if not df.empty and df["session_id"].iloc[0] != session_id:
            buffer_table.delete(f"session_id != '{session_id}'")
        row = {
            "session_id": str(session_id) if session_id is not None else "",
            "turn_id": int(turn_id) if turn_id is not None else 0,
            "user_query": str(user_query) if user_query is not None else "",
            "bot_response": str(bot_response) if bot_response is not None else "",
            "timestamp": self._get_timestamp()
        }
        buffer_table.add([row])

    def log_to_items(self, session_id, turn_id, user_query, bot_response, summary, metrics, retrieval_type,llm_model, top_k_dense, top_k_sparse, top_k_combined):
        items_table = self.db.open_table("items")
        def safe_int(val, default=0):
            try:
                if val is None:
                    return default
                return int(float(val))
            except Exception:
                return default
        
        row = {
            "session_id": str(session_id) if session_id is not None else "",
            "turn_id": int(turn_id) if turn_id is not None else 0,
            "user_query": str(user_query) if user_query is not None else "",
            "bot_response": str(bot_response) if bot_response is not None else "",
            "summary": str(summary) if summary is not None else "",
            "llm_model": str(llm_model) if llm_model is not None else "",
            "top_k_dense": safe_int(top_k_dense),
            "top_k_sparse": safe_int(top_k_sparse),
            "top_k_combined": safe_int(top_k_combined),
            "intent_labels": ",".join(metrics.get("intent_labels", [])),
            "intent_confidence": float(metrics.get("intent_confidence", 0)),
            "slots": json.dumps(metrics.get("slots", {})),
            "retrieval_type": str(retrieval_type) if retrieval_type else "",
            "retrieval_strength": float(metrics.get("retrieval_strength", 0)),
            "retrieved_chunk_ids": ",".join(str(x) for x in metrics.get("retrieved_chunk_ids", [])),
            "used_rag": bool(metrics.get("used_rag", False)),
            "answer_decision": str(metrics.get("answer_decision", "")),
            "embedding_latency_ms": safe_int(metrics.get("embedding_time", 0) * 1000),
            "dense_retrieval_latency_ms": safe_int(metrics.get("dense_search_time", 0) * 1000),
            "sparse_retrieval_latency_ms": safe_int(metrics.get("sparse_search_time", 0) * 1000),
            "llm_latency_ms": safe_int(metrics.get("generation_time", 0) * 1000),
            "total_latency_ms": safe_int(metrics.get("total_time", 0) * 1000),
            "timestamp": self._get_timestamp(),
            # RAGAS
            "golden_answer": None,
            "context_recall": None,
            "context_precision": None,
            "answer_correctness": None,
            "faithfulness": None,
            "answer_similarity": None,
            # LLM COST/TOKENS
            "router_input_tokens": int(metrics.get("router_input_tokens", 0)),
            "router_output_tokens": int(metrics.get("router_output_tokens", 0)),
            "router_total_tokens": int(metrics.get("router_total_tokens", 0)),
            "router_cost": float(metrics.get("router_cost", 0)),
            "summarizer_input_tokens": int(metrics.get("summarizer_input_tokens", 0)),
            "summarizer_output_tokens": int(metrics.get("summarizer_output_tokens", 0)),
            "summarizer_total_tokens": int(metrics.get("summarizer_total_tokens", 0)),
            "summarizer_cost": float(metrics.get("summarizer_cost", 0)),
            "main_input_tokens": int(metrics.get("main_input_tokens", 0)),
            "main_output_tokens": int(metrics.get("main_output_tokens", 0)),
            "main_total_tokens": int(metrics.get("main_total_tokens", 0)),
            "main_llm_cost": float(metrics.get("main_llm_cost", 0)),
            "total_turn_cost": float(metrics.get("total_turn_cost", 0)),
        }
        items_table.add([row])



    def log_session_summary(self, session_id, llm_model, top_k_dense, top_k_sparse, top_k_combined, retrieval_type):
        items_df = self.db.open_table("items").to_pandas()
        session_df = items_df[items_df["session_id"] == session_id]
        if session_df.empty:
            return
        created_at = session_df["timestamp"].min()
        ended_at = session_df["timestamp"].max()
        total_turns = len(session_df)
        total_time = (pd.to_datetime(ended_at) - pd.to_datetime(created_at)).total_seconds() * 1000
        total_cost = float(session_df["total_turn_cost"].sum() if "total_turn_cost" in session_df.columns else 0.0)
        total_input_tokens= int(session_df["router_input_tokens"].sum() + session_df["summarizer_input_tokens"].sum() + session_df["main_input_tokens"].sum())
        total_output_tokens= int( session_df["router_output_tokens"].sum() + session_df["summarizer_output_tokens"].sum() + session_df["main_output_tokens"].sum())
        total_tokens=total_input_tokens + total_output_tokens
        summary = {
            "session_id": str(session_id),
            "total_turns": int(total_turns),
            "total_session_time_ms": int(total_time),
            "llm_model": str(llm_model) if llm_model is not None else "",
            "top_k_dense": int(top_k_dense) if top_k_dense is not None else 0,
            "top_k_sparse": int(top_k_sparse) if top_k_sparse is not None else 0,
            "top_k_combined": int(top_k_combined) if top_k_combined is not None else 0,
            "retrieval_type": str(retrieval_type) if retrieval_type else "",
            "avg_embedding_latency_ms": float(session_df["embedding_latency_ms"].mean()) if not session_df["embedding_latency_ms"].isnull().all() else 0.0,
            "avg_dense_retrieval_latency_ms": float(session_df["dense_retrieval_latency_ms"].mean()) if not session_df["dense_retrieval_latency_ms"].isnull().all() else 0.0,
            "avg_sparse_retrieval_latency_ms": float(session_df["sparse_retrieval_latency_ms"].mean()) if not session_df["sparse_retrieval_latency_ms"].isnull().all() else 0.0,
            "avg_llm_latency_ms": float(session_df["llm_latency_ms"].mean()) if not session_df["llm_latency_ms"].isnull().all() else 0.0,
            "avg_retrieval_strength": float(session_df["retrieval_strength"].mean()) if "retrieval_strength" in session_df.columns and not session_df["retrieval_strength"].isnull().all() else 0.0,
            "created_at":self._get_timestamp(),
            "ended_at": self._get_timestamp(),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
        }
        sessions_table = self.db.open_table("sessions")
        sessions_table.delete(f"session_id == '{session_id}'")
        sessions_table.add([summary])

    def update_ground_truth(self, session_id: str, turn_id: int, ground_truth: str) -> bool:
        try:
            items_table = self.db.open_table("items")
            df = items_table.to_pandas()
            
            if df.empty:
                print(f" Items table is empty")
                return False
            
            # Find the specific row to update
            mask = (df["session_id"] == str(session_id)) & (df["turn_id"] == int(turn_id))
            
            if not mask.any():
                print(f" No item found for session_id: {session_id}, turn_id: {turn_id}")
                available_ids = df[['session_id', 'turn_id']].head(5).to_dict('records')
                print(f" Available session_ids and turn_ids (first 5): {available_ids}")
                return False
            
            # Update the golden_answer field
            df.loc[mask, "golden_answer"] = str(ground_truth)
            
            # Replace the entire table with updated data
            items_table.delete("session_id IS NOT NULL")  # Clear all data
            items_table.add(df.to_dict('records'))
            
            print(f" Successfully updated ground truth for {session_id}_{turn_id}")
            return True
            
        except Exception as e:
            print(f" Error updating ground truth: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_ragas_scores(self, session_id, turn_id, ragas_scores, golden_answer=None):
        """Update RAGAS evaluation scores for a specific item"""
        items_table = self.db.open_table("items")
        
        # Get the current data
        df = items_table.to_pandas()
        
        # Find the specific row to update
        mask = (df["session_id"] == str(session_id)) & (df["turn_id"] == int(turn_id))
        
        if mask.any():
            # Update the row with RAGAS scores
            if golden_answer is not None:
                df.loc[mask, "golden_answer"] = str(golden_answer)
            
            # Only update scores that are provided
            for metric, score in ragas_scores.items():
                if metric in ["context_recall", "context_precision", "answer_correctness", "faithfulness", "answer_similarity"]:
                    df.loc[mask, metric] = float(score) if score is not None else None
            
            # Replace the entire table with updated data
            items_table.delete("session_id IS NOT NULL")  # Clear all data
            items_table.add(df.to_dict('records'))
            
            return True
        return False

    def bulk_update_ragas_scores(self, evaluation_results):
        """Bulk update RAGAS scores for multiple items"""
        items_table = self.db.open_table("items")
        df = items_table.to_pandas()
        
        updated_count = 0
        for result in evaluation_results:
            session_id = result.get("session_id")
            turn_id = result.get("turn_id")
            scores = result.get("scores", {})
            golden_answer = result.get("golden_answer")
            
            mask = (df["session_id"] == str(session_id)) & (df["turn_id"] == int(turn_id))
            
            if mask.any():
                # Update golden answer if provided
                if golden_answer is not None and golden_answer != "":
                    df.loc[mask, "golden_answer"] = str(golden_answer)
                
                # Update RAGAS scores
                for metric, score in scores.items():
                    if metric in ["context_recall", "context_precision", "answer_correctness", "faithfulness", "answer_similarity"]:
                        df.loc[mask, metric] = float(score) if score is not None else None
                
                updated_count += 1
        
        if updated_count > 0:
            # Replace the entire table with updated data
            items_table.delete("session_id IS NOT NULL")
            items_table.add(df.to_dict('records'))
        
        return updated_count

    def bulk_update_ground_truth(self, ground_truth_updates):
        
        try:
            items_table = self.db.open_table("items")
            df = items_table.to_pandas()
            
            if df.empty:
                print(f" Items table is empty for bulk update")
                return 0
            
            updated_count = 0
            for update in ground_truth_updates:
                session_id = update.get("session_id")
                turn_id = update.get("turn_id")
                ground_truth = update.get("ground_truth", "")
                
                if not ground_truth.strip():
                    continue
                
                mask = (df["session_id"] == str(session_id)) & (df["turn_id"] == int(turn_id))
                
                if mask.any():
                    df.loc[mask, "golden_answer"] = str(ground_truth)
                    updated_count += 1
            
            if updated_count > 0:
                # Replace the entire table with updated data
                items_table.delete("session_id IS NOT NULL")
                items_table.add(df.to_dict('records'))
                print(f" Bulk updated {updated_count} ground truth entries")
            
            return updated_count
            
        except Exception as e:
            print(f" Error in bulk ground truth update: {e}")
            return 0

    def get_items_with_ground_truth(self):
        """
        Get items that have ground truth (golden_answer is not null/empty)
        """
        df = self.db.open_table("items").to_pandas()
        if not df.empty:
            # Filter items where golden_answer is not null and not empty
            has_gt = df["golden_answer"].notna() & (df["golden_answer"] != "") & (df["golden_answer"] != "None")
            return df[has_gt]
        return pd.DataFrame()

    def get_items_without_ground_truth(self):
        """
        Get items that don't have ground truth yet
        """
        df = self.db.open_table("items").to_pandas()
        if not df.empty:
            # Filter items where golden_answer is null, empty, or "None"
            no_gt = df["golden_answer"].isna() | (df["golden_answer"] == "") | (df["golden_answer"] == "None")
            return df[no_gt]
        return pd.DataFrame()

    def get_ground_truth_stats(self):
        """
        Get statistics about ground truth coverage
        """
        try:
            df = self.db.open_table("items").to_pandas()
            if df.empty:
                return {
                    "total_items": 0,
                    "with_ground_truth": 0,
                    "without_ground_truth": 0,
                    "coverage_percentage": 0.0
                }
            
            total_items = len(df)
            with_gt = len(self.get_items_with_ground_truth())
            without_gt = total_items - with_gt
            coverage = (with_gt / total_items * 100) if total_items > 0 else 0.0
            
            return {
                "total_items": total_items,
                "with_ground_truth": with_gt,
                "without_ground_truth": without_gt,
                "coverage_percentage": round(coverage, 2)
            }
        except Exception as e:
            print(f" Error getting ground truth stats: {e}")
            return {
                "total_items": 0,
                "with_ground_truth": 0,
                "without_ground_truth": 0,
                "coverage_percentage": 0.0
            }

    def debug_items_table(self):
        """
        Debug method to inspect items table structure and data
        """
        try:
            items_table = self.db.open_table("items")
            df = items_table.to_pandas()
            
            print(f" Items Table Debug Info:")
            print(f"   - Total rows: {len(df)}")
            print(f"   - Columns: {list(df.columns)}")
            
            if not df.empty:
                print(f"   - Sample session_ids: {df['session_id'].head(3).tolist()}")
                print(f"   - Sample turn_ids: {df['turn_id'].head(3).tolist()}")
                print(f"   - Golden answer column exists: {'golden_answer' in df.columns}")
                
                if 'golden_answer' in df.columns:
                    gt_count = df['golden_answer'].notna().sum()
                    non_empty_gt = ((df['golden_answer'].notna()) & (df['golden_answer'] != "") & (df['golden_answer'] != "None")).sum()
                    print(f"   - Items with golden_answer (not null): {gt_count}")
                    print(f"   - Items with non-empty golden_answer: {non_empty_gt}")
                    
                    # Show sample data
                    print(f"   - Sample rows:")
                    sample_df = df[['session_id', 'turn_id', 'user_query', 'golden_answer']].head(3)
                    for idx, row in sample_df.iterrows():
                        print(f"     Row {idx}: session_id={row['session_id']}, turn_id={row['turn_id']}, query='{row['user_query'][:50]}...', gt='{row['golden_answer']}'")
            
            return df
            
        except Exception as e:
            print(f" Error debugging items table: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    

    def get_buffer(self, session_id):
        df = self.db.open_table("buffer").to_pandas()
        return df[df["session_id"] == session_id] if not df.empty else pd.DataFrame()

    def get_items(self, session_id):
        df = self.db.open_table("items").to_pandas()
        return df[df["session_id"] == session_id] if not df.empty else pd.DataFrame()
    
    def get_items_full(self):
        df = self.db.open_table("items").to_pandas()
        return df if not df.empty else pd.DataFrame()

    def get_all_sessions(self):
        df = self.db.open_table("sessions").to_pandas()
        return df if not df.empty else pd.DataFrame()
    
    def get_curr_session(self, session_id):
        df = self.db.open_table("sessions").to_pandas()
        return df[df["session_id"] == session_id] if not df.empty else pd.DataFrame()
    
    def get_ragas_evaluated_items(self):
        """Get items that have RAGAS evaluation scores (not null)"""
        df = self.db.open_table("items").to_pandas()
        if not df.empty:
            # Filter items where at least one RAGAS score is not null
            ragas_cols = ["context_recall", "context_precision", "answer_correctness", "faithfulness", "answer_similarity"]
            has_ragas = df[ragas_cols].notna().any(axis=1)
            return df[has_ragas]
        return pd.DataFrame()
    
    def get_items_without_ragas(self):
        """Get items that don't have RAGAS evaluation scores yet"""
        df = self.db.open_table("items").to_pandas()
        if not df.empty:
            # Filter items where all RAGAS scores are null
            ragas_cols = ["context_recall", "context_precision", "answer_correctness", "faithfulness", "answer_similarity"]
            no_ragas = df[ragas_cols].isna().all(axis=1)
            return df[no_ragas]
        return pd.DataFrame()