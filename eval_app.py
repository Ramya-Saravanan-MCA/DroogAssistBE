import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import asyncio
from typing import List, Dict


from dotenv import load_dotenv
load_dotenv()
# RAGAS imports with model configuration
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy, 
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from utils.secrets_manager import load_secrets_into_env
load_secrets_into_env()


from config import GROQ_API_KEY, OPENAI_API_KEY, EMBEDDING_MODEL_NAME, LANCEDB_PATH
 
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Langchain imports for model setup
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from db.session_logger import SessionLogger

# Page config
st.set_page_config(
    page_title="RAGAS Evaluation Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title(" RAGAS Evaluation Dashboard")


# Initialize session state
for key in ['evaluation_data', 'annotated_data', 'ragas_results', 'models_configured']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'models_configured' else False

# Sidebar configuration
st.sidebar.header("Configuration")
chatdb_path = os.getenv("CHATDB_PATH", "s3://droogbucket/lancedb/chatdb")
lancedb_path = os.getenv("LANCEDB_PATH", "s3://droogbucket/lancedb")

# Model configuration
st.sidebar.header("Model Configuration")
openai_api_key = os.getenv("OPENAI_API_KEY", "")
llm_model = st.sidebar.selectbox("LLM Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=0)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# RAGAS metrics selection
st.sidebar.header("RAGAS Metrics")
selected_metrics = st.sidebar.multiselect(
    "Select metrics to evaluate:",
    options=[
        "faithfulness", 
        "context_precision",
        "context_recall",
        "answer_correctness",
        "answer_similarity"
    ],
    default=["faithfulness", "answer_correctness", "context_precision", "context_recall","answer_similarity"]
)

def configure_ragas_models(openai_api_key: str, llm_model: str, embedding_model: str):
    """Configure RAGAS with specified models"""
    try:
        # Configure LLM (GPT-3.5-turbo)
        llm = ChatOpenAI(
            model=llm_model,
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        
        # Configure Embeddings (SentenceTransformer)
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
        # Wrap models for RAGAS
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        return ragas_llm, ragas_embeddings
        
    except Exception as e:
        st.error(f"Error configuring models: {e}")
        return None, None

def parse_ragas_results(results):
    """Parse RAGAS evaluation results properly"""
    try:
        # Convert to pandas DataFrame
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
        else:
            # Handle different RAGAS result formats
            df = pd.DataFrame(results)
        
        # Extract metric scores (mean values)
        scores = {}
        for col in df.columns:
            if col not in ['question', 'answer', 'contexts', 'ground_truth']:
                if pd.api.types.is_numeric_dtype(df[col]):
                    scores[col] = float(df[col].mean())
        
        return scores, df
        
    except Exception as e:
        st.error(f"Error parsing RAGAS results: {e}")
        return {}, pd.DataFrame()

class RAGASEvaluator:
    def __init__(self, chatdb_path: str, lancedb_path: str):
        self.chatdb_path = chatdb_path
        self.lancedb_path = lancedb_path
        self.session_logger = SessionLogger(chatdb_path)
        self.ragas_llm = None
        self.ragas_embeddings = None
    
    def configure_models(self, openai_api_key: str, llm_model: str, embedding_model: str):
        """Configure the models for RAGAS evaluation"""
        self.ragas_llm, self.ragas_embeddings = configure_ragas_models(
            openai_api_key, llm_model, embedding_model
        )
        return self.ragas_llm is not None and self.ragas_embeddings is not None
    
    def extract_evaluation_data(self, session_ids: List[str] = None) -> pd.DataFrame:
        """Extract evaluation data from items table"""
        if session_ids:
            all_items = []
            for session_id in session_ids:
                items_df = self.session_logger.get_items(session_id)
                all_items.append(items_df)
            eval_df = pd.concat(all_items, ignore_index=True) if all_items else pd.DataFrame()
        else:
            eval_df = self.session_logger.get_items_full()
        
        return eval_df
    
    def prepare_ragas_dataset(self, eval_df: pd.DataFrame) -> List[Dict]:
        """Convert items table data to RAGAS-compatible format - Enhanced with existing ground truth"""
        import lancedb
        
        # Load knowledge base for chunk text mapping
        try:
            db = lancedb.connect(self.lancedb_path)
            table = db.open_table("unified_knowledge_base")
            kb_df = table.to_pandas()
            chunk_text_map = dict(zip(kb_df['chunk_id'].astype(str), kb_df['text']))
        except Exception as e:
            st.error(f"Could not load knowledge base: {e}")
            chunk_text_map = {}
        
        ragas_data = []
        for idx, row in eval_df.iterrows():
            # Parse retrieved chunk IDs
            retrieved_ids = []
            if pd.notna(row.get("retrieved_chunk_ids", "")):
                retrieved_ids = [
                    cid.strip() for cid in str(row["retrieved_chunk_ids"]).split(",") 
                    if cid.strip() and cid.strip().isdigit()
                ]
            
            # Map to chunk texts (contexts)
            contexts = []
            for chunk_id in retrieved_ids:
                if chunk_id in chunk_text_map:
                    contexts.append(chunk_text_map[chunk_id])
            
            # Skip if no contexts retrieved or no RAG used
            if not contexts or not row.get("used_rag", False):
                continue
            
            # Extract existing ground truth from database - ENHANCED
            existing_ground_truth = ""
            if pd.notna(row.get("golden_answer", "")):
                existing_ground_truth = str(row["golden_answer"])
            
            ragas_record = {
                "question": row.get("user_query", ""),
                "answer": row.get("bot_response", ""),
                "contexts": contexts,
                "ground_truth": existing_ground_truth,  # Load existing from DB
                "has_existing_gt": bool(existing_ground_truth.strip()),  # Flag for UI
                
                # Metadata
                "query_id": f"{row.get('session_id', 'unknown')}_{row.get('turn_id', idx)}",
                "session_id": row.get("session_id", ""),
                "turn_id": row.get("turn_id", 0),
                "retrieval_strength": row.get("retrieval_strength", 0),
                "total_latency_ms": row.get("total_latency_ms", 0),
                "timestamp": row.get("timestamp", "")
            }
            
            ragas_data.append(ragas_record)
        
        return ragas_data
    
    def save_ground_truth_to_db(self, session_id: str, turn_id: int, ground_truth: str):

        try:
            success = self.session_logger.update_ground_truth(session_id, turn_id, ground_truth)
            return success
        except Exception as e:
            st.error(f"Error saving ground truth to database: {e}")
            return False
    
    def get_metric_objects(self, metric_names: List[str]):
        """Get RAGAS metric objects with configured models"""
        # Configure metrics with custom models
        metric_objects = []
        
        for metric_name in metric_names:
            if metric_name == "faithfulness":
                metric = faithfulness
                if self.ragas_llm:
                    metric.llm = self.ragas_llm
                metric_objects.append(metric)
                
            elif metric_name == "context_precision":
                metric = context_precision
                if self.ragas_llm:
                    metric.llm = self.ragas_llm
                metric_objects.append(metric)
                
            elif metric_name == "context_recall":
                metric = context_recall
                if self.ragas_llm and self.ragas_embeddings:
                    metric.llm = self.ragas_llm
                    metric.embeddings = self.ragas_embeddings
                metric_objects.append(metric)
                
            elif metric_name == "answer_correctness":
                metric = answer_correctness
                if self.ragas_llm and self.ragas_embeddings:
                    metric.llm = self.ragas_llm
                    metric.embeddings = self.ragas_embeddings
                metric_objects.append(metric)
                
            elif metric_name == "answer_similarity":
                metric = answer_similarity
                if self.ragas_embeddings:
                    metric.embeddings = self.ragas_embeddings
                metric_objects.append(metric)
        
        return metric_objects

# Model configuration check
if openai_api_key:
    if st.sidebar.button("Configure Models"):
        with st.spinner("Configuring RAGAS models..."):
            evaluator = RAGASEvaluator(chatdb_path, lancedb_path)
            success = evaluator.configure_models(openai_api_key, llm_model, embedding_model)
            if success:
                st.session_state.models_configured = True
                st.session_state.evaluator = evaluator
                st.sidebar.success(f" Models configured: {llm_model} + {embedding_model}")
            else:
                st.sidebar.error(" Failed to configure models")
else:
    st.sidebar.warning(" Please provide OpenAI API key")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Overview", 
    " Data Extraction", 
    " Annotation", 
    " RAGAS Evaluation",
    " Results & Export"
])

with tab1:
    st.header("RAGAS Evaluation Overview")
    
    # Model status
    if st.session_state.models_configured:
        st.success(f" Models configured: {llm_model} + {embedding_model}")
    else:
        st.warning(" Please configure models in the sidebar first")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            session_logger = SessionLogger(chatdb_path)
            items_df = session_logger.get_items_full()
            rag_queries = len(items_df[items_df['used_rag'] == True]) if not items_df.empty else 0
            st.metric("Total RAG Queries", rag_queries)
        except:
            st.metric("Total RAG Queries", "N/A")
    
    with col2:
        st.metric("Selected Metrics", len(selected_metrics))
    
    with col3:
        status = " Ready" if st.session_state.ragas_results else " Pending"
        st.metric("Evaluation Status", status)
    
    st.markdown(f"""

    ### RAGAS Metrics:
    
    **Retrieval Metrics:**
    -  **Context Precision**: Measures how relevant retrieved contexts are
    -  **Context Recall**: Measures if all relevant info was retrieved

    **Generation Metrics:**
    -  **Faithfulness**: Measures if answer is grounded in retrieved context
    -  **Answer Correctness**: Measures factual accuracy (needs ground truth)
    -  **Answer Similarity**: Measures semantic similarity (needs ground truth)
    """)
    
    # Display Last 10 Items and Download Entire Table
    st.subheader("Items Table Overview")
    
    try:
        session_logger = SessionLogger(chatdb_path)
        items_df = session_logger.get_items_full()
        
        if not items_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Items", len(items_df))
            
            with col2:
                # Download entire table as CSV
                csv_data = items_df.to_csv(index=False)
                st.download_button(
                    " Download Entire Items Table",
                    data=csv_data,
                    file_name=f"items_table_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help=f"Download all {len(items_df)} items from the database"
                )
            
            # Display last 10 items
            st.markdown("**Last 10 Items Added:**")
            last_10_items = items_df.tail(10)
            
            # Format for display
            display_cols = ['session_id', 'turn_id', 'user_query', 'bot_response', 'used_rag', 'retrieved_chunks','answer_decision', 'golden_answer','context_recall','context_precision','answer_similiarity','answer_correctness','faithfulness', 'timestamp']
            available_cols = [col for col in display_cols if col in last_10_items.columns]
            
            display_df = last_10_items[available_cols].copy()
            
            # Truncate long text for display
            if 'user_query' in display_df.columns:
                display_df['user_query'] = display_df['user_query'].apply(lambda x: str(x)[:60] + "..." if len(str(x)) > 60 else str(x))
            if 'bot_response' in display_df.columns:
                display_df['bot_response'] = display_df['bot_response'].apply(lambda x: str(x)[:60] + "..." if len(str(x)) > 60 else str(x))
            if 'golden_answer' in display_df.columns:
                display_df['golden_answer'] = display_df['golden_answer'].apply(
                    lambda x: " Yes" if pd.notna(x) and str(x).strip() and str(x) != "None" else " No"
                )
            
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.info("No items found in the database.")
            
    except Exception as e:
        st.error(f"Error accessing items table: {str(e)}")

with tab2:
    st.header(" Data Extraction for RAGAS")
    
    if not st.session_state.models_configured:
        st.warning(" Please configure models first in the sidebar")
    else:
        try:
            evaluator = st.session_state.evaluator
            session_logger = SessionLogger(chatdb_path)
            sessions_df = session_logger.get_all_sessions()
            
            if not sessions_df.empty:
                # Session selection
                session_options = ["All Sessions"] + list(sessions_df['session_id'].unique())
                selected_sessions = st.multiselect(
                    "Select Sessions to Evaluate",
                    options=session_options,
                    default=["All Sessions"]
                )
                
                # Filter options
                #col1 = st.columns(1)
                #with col1:
                    #min_retrieval_strength = st.slider("Minimum Retrieval strength", min_value=0.016, max_value=0.032, value=0.025, step=0.001)
               # with col1:
                only_rag_queries = st.checkbox("Only RAG queries", value=True)
                
                # Extract data button
                if st.button("Extract RAGAS Dataset"):
                    with st.spinner("Extracting data from items table..."):
                        if "All Sessions" in selected_sessions:
                            items_df = evaluator.extract_evaluation_data()
                        else:
                            session_ids = [s for s in selected_sessions if s != "All Sessions"]
                            items_df = evaluator.extract_evaluation_data(session_ids)
                        
                        # Apply filters
                        if only_rag_queries:
                            items_df = items_df[items_df['used_rag'] == True]
                        
                        #items_df = items_df[items_df['retrieval_strength'] >= min_retrieval_strength]
                        
                        ragas_data = evaluator.prepare_ragas_dataset(items_df)
                        st.session_state.evaluation_data = ragas_data
                        
                        # Count existing ground truths
                        existing_gt_count = sum(1 for item in ragas_data if item['has_existing_gt'])
                        
                        st.success(f" Extracted {len(ragas_data)} queries for RAGAS evaluation")
                        st.info(f" Found {existing_gt_count} queries with existing ground truth in database")
                
                # Display extracted data with ground truth status
                if st.session_state.evaluation_data:
                    st.subheader("RAGAS Dataset Preview")
                    
                    preview_data = []
                    for item in st.session_state.evaluation_data[:10]:
                        preview_data.append({
                            "Query ID": item["query_id"],
                            "Question": item["question"][:60] + "..." if len(item["question"]) > 60 else item["question"],
                            "Answer": item["answer"][:60] + "..." if len(item["answer"]) > 60 else item["answer"],
                            "Contexts": len(item["contexts"]),
                            "Has Ground Truth": " Yes" if item["has_existing_gt"] else " No",
                            "Retrieval Strength": f"{item['retrieval_strength']:.3f}"
                        })
                    
                    total_queries = len(st.session_state.evaluation_data)
                    existing_gt_count = sum(1 for item in st.session_state.evaluation_data if item['has_existing_gt'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Queries", total_queries)
                    with col2:
                        st.metric("With Ground Truth", existing_gt_count)
                    with col3:
                        st.metric("Need Annotation", total_queries - existing_gt_count)
                    
                    st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
                    
                    # View Individual Questions Instead of Just First Item
                    st.subheader(" View Individual Questions")
                    
                    # Question selector
                    total_questions = len(st.session_state.evaluation_data)
                    if total_questions > 0:
                        question_options = []
                        for i, item in enumerate(st.session_state.evaluation_data):
                            gt_status = "✅" if item['has_existing_gt'] else "❌"
                            question_options.append(f"Question {i+1} : {item['question'][:50]}...")
                        
                        selected_question_idx = st.selectbox(
                            f"Select Question to View (1-{total_questions}):",
                            range(total_questions),
                            format_func=lambda x: question_options[x]
                        )
                        
                        # Display selected question details  with context 
                        selected_item = st.session_state.evaluation_data[selected_question_idx]
                        
                        with st.expander(f" View Question {selected_question_idx + 1} Details", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("** Question:**")
                                st.text_area("", value=selected_item["question"], height=100, disabled=True, key=f"q_view_{selected_question_idx}")
                                
                                
                            
                            with col2:
                                st.markdown("** Generated Answer:**")
                                st.text_area("", value=selected_item["answer"], height=100, disabled=True, key=f"a_view_{selected_question_idx}")
                                
                                if selected_item["has_existing_gt"]:
                                    st.markdown("** Ground Truth:**")
                                    st.text_area("", value=selected_item["ground_truth"], height=100, disabled=True, key=f"gt_view_{selected_question_idx}")
                                else:
                                    st.markdown("** No Ground Truth Available**")
                            
                            # Retrieved contexts
                            st.markdown("** Retrieved Contexts:**")
                            for i, context in enumerate(selected_item["contexts"]):
                                with st.expander(f"Context {i+1}"):
                                    st.text(context[:800] + "..." if len(context) > 800 else context)
                    
            else:
                st.warning("No sessions found in the database.")
                
        except Exception as e:
            st.error(f"Error accessing session data: {str(e)}")

with tab3:
    st.header(" Ground Truth Annotation")
    
    if st.session_state.evaluation_data:
        st.info("💡 **Note**: Ground truth is only needed for answer_correctness and answer_similarity metrics")
        
        # Check which metrics need ground truth
        needs_ground_truth = any(metric in selected_metrics for metric in ["answer_correctness", "answer_similarity"])
        
        # Count existing ground truths
        existing_gt_count = sum(1 for item in st.session_state.evaluation_data if item['has_existing_gt'])
        total_queries = len(st.session_state.evaluation_data)
        
        # Display ground truth status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("With Existing GT", existing_gt_count)
        with col3:
            st.metric("Need Annotation", total_queries - existing_gt_count)
        
        if not needs_ground_truth:
            st.success(" Selected metrics don't require ground truth annotation!")
        else:
            st.warning(" Selected metrics require ground truth annotation")
            
        
            annotation_mode = st.radio(
                "Choose annotation method:",
                ["Smart Annotation ", "CSV Upload"]
            )
            
            if annotation_mode == "Smart Annotation ":
                st.subheader( "Ground Truth Annotation")
                
                # Filter options
                filter_option = st.selectbox(
                    "Filter queries to annotate:",
                    ["All Queries", "Only Missing Ground Truth", "Only With Ground Truth (Edit)", "Specific Query"]
                )
                
                # Apply filter
                if filter_option == "Only Missing Ground Truth":
                    filtered_queries = [item for item in st.session_state.evaluation_data if not item['has_existing_gt']]
                elif filter_option == "Only With Ground Truth (Edit)":
                    filtered_queries = [item for item in st.session_state.evaluation_data if item['has_existing_gt']]
                else:
                    filtered_queries = st.session_state.evaluation_data
                
                if not filtered_queries:
                    st.info(f"No queries found for filter: {filter_option}")
                else:
                    # Query selector
                    query_options = []
                    for item in filtered_queries:
                        gt_status = "✅" if item['has_existing_gt'] else "❌"
                        query_options.append(f" {item['query_id']}: {item['question'][:50]}...")
                    
                    selected_idx = st.selectbox(
                        f"Select Query to Annotate ({len(filtered_queries)} available):",
                        range(len(query_options)),
                        format_func=lambda x: query_options[x]
                    )
                    
                    current_item = filtered_queries[selected_idx]
                    
                    # Display current query info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Question:**")
                        st.text_area("", value=current_item["question"], height=100, disabled=True, key="question_display")
                        
                        st.markdown("**Retrieved Contexts:**")
                        for i, context in enumerate(current_item["contexts"]):
                            with st.expander(f"Context {i+1}"):
                                st.text(context[:500] + "..." if len(context) > 500 else context)
                    
                    with col2:
                        st.markdown("**Generated Answer:**")
                        st.text_area("", value=current_item["answer"], height=100, disabled=True, key="answer_display")
                        
                        # Ground truth annotation with existing value detection
                        if current_item['has_existing_gt']:
                            st.success(f" Existing Ground Truth Found")
                            st.info(f"**Current GT:** {current_item['ground_truth'][:100]}{'...' if len(current_item['ground_truth']) > 100 else ''}")
                            
                            # Option to use existing or update
                            gt_action = st.radio(
                                "Ground Truth Action:",
                                ["Use Existing", "Update/Edit", "Replace Completely"],
                                key=f"gt_action_{selected_idx}"
                            )
                            
                            if gt_action == "Use Existing":
                                st.info(" Using existing ground truth from database")
                                if st.button(" Confirm Use Existing", key=f"confirm_existing_{selected_idx}"):
                                    st.success(" Existing ground truth confirmed!")
                                    
                            elif gt_action == "Update/Edit":
                                st.markdown("**Edit Ground Truth:**")
                                ground_truth = st.text_area(
                                    "Edit the ground truth answer:",
                                    value=current_item["ground_truth"],
                                    height=150,
                                    key=f"gt_edit_{selected_idx}",
                                    help="Modify the existing ground truth answer"
                                )
                                
                                if st.button(" Update Ground Truth in Database", key=f"update_gt_{selected_idx}"):
                                    if ground_truth.strip():
                                        # Save to database immediately
                                        evaluator = st.session_state.evaluator
                                        success = evaluator.save_ground_truth_to_db(
                                            current_item["session_id"], 
                                            current_item["turn_id"], 
                                            ground_truth
                                        )
                                        
                                        if success:
                                            # Update evaluation data in memory
                                            for i, item in enumerate(st.session_state.evaluation_data):
                                                if item['query_id'] == current_item['query_id']:
                                                    st.session_state.evaluation_data[i]["ground_truth"] = ground_truth
                                                    st.session_state.evaluation_data[i]["has_existing_gt"] = True
                                                    break
                                            
                                            st.success(" Ground truth updated in database successfully!")
                                            st.info(f" Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                            st.rerun()
                                        else:
                                            st.error(" Failed to update ground truth in database")
                                    else:
                                        st.warning(" Please enter ground truth before saving")
                                        
                            else:  # Replace Completely
                                st.markdown("**Replace Ground Truth:**")
                                ground_truth = st.text_area(
                                    "Enter new ground truth answer:",
                                    value="",
                                    height=150,
                                    key=f"gt_replace_{selected_idx}",
                                    help="Provide a completely new ground truth answer"
                                )
                                
                                if st.button(" Replace Ground Truth in Database", key=f"replace_gt_{selected_idx}"):
                                    if ground_truth.strip():
                                        # Save to database immediately
                                        evaluator = st.session_state.evaluator
                                        success = evaluator.save_ground_truth_to_db(
                                            current_item["session_id"], 
                                            current_item["turn_id"], 
                                            ground_truth
                                        )
                                        
                                        if success:
                                            # Update evaluation data in memory
                                            for i, item in enumerate(st.session_state.evaluation_data):
                                                if item['query_id'] == current_item['query_id']:
                                                    st.session_state.evaluation_data[i]["ground_truth"] = ground_truth
                                                    st.session_state.evaluation_data[i]["has_existing_gt"] = True
                                                    break
                                            
                                            st.success(" Ground truth replaced in database successfully!")
                    
                                            st.rerun()
                                        else:
                                            st.error(" Failed to replace ground truth in database")
                                    else:
                                        st.warning(" Please enter ground truth before saving")
                        
                        else:
                            # No existing ground truth - create new
                            st.warning(" No Ground Truth Found")
                            st.markdown("**Create New Ground Truth:**")
                            ground_truth = st.text_area(
                                "Enter ground truth answer:",
                                value="",
                                height=150,
                                key=f"gt_new_{selected_idx}",
                                help="Provide the ideal/correct answer for this question"
                            )
                            
                            if st.button(" Save New Ground Truth to Database", key=f"save_new_gt_{selected_idx}"):
                                if ground_truth.strip():
                                    # Save to database immediately
                                    evaluator = st.session_state.evaluator
                                    success = evaluator.save_ground_truth_to_db(
                                        current_item["session_id"], 
                                        current_item["turn_id"], 
                                        ground_truth
                                    )
                                    
                                    if success:
                                        # Update evaluation data in memory
                                        for i, item in enumerate(st.session_state.evaluation_data):
                                            if item['query_id'] == current_item['query_id']:
                                                st.session_state.evaluation_data[i]["ground_truth"] = ground_truth
                                                st.session_state.evaluation_data[i]["has_existing_gt"] = True
                                                break
                                        
                                        st.success(" Ground truth saved to database successfully!")

                                        st.rerun()
                                    else:
                                        st.error(" Failed to save ground truth to database")
                                else:
                                    st.warning(" Please enter ground truth before saving")
            
            elif annotation_mode == "CSV Upload":
                st.subheader(" Upload Ground Truth CSV")
                
                # Create template
                if st.button("Generate CSV Template"):
                    template_data = []
                    for item in st.session_state.evaluation_data:
                        template_data.append({
                            "query_id": item["query_id"],
                            "session_id": item["session_id"],
                            "turn_id": item["turn_id"],
                            "question": item["question"],
                            "generated_answer": item["answer"],
                            "existing_ground_truth": item.get("ground_truth", ""),
                            "ground_truth": item.get("ground_truth", "")  # To be filled/edited
                        })
                    
                    template_df = pd.DataFrame(template_data)
                    csv_data = template_df.to_csv(index=False)
                    
                    st.download_button(
                        " Download CSV Template",
                        data=csv_data,
                        file_name=f"ragas_ground_truth_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Upload annotated CSV with immediate database save
                uploaded_file = st.file_uploader(
                    "Upload your annotated CSV file",
                    type=['csv'],
                    key="gt_upload"
                )
                
                if uploaded_file:
                    try:
                        gt_df = pd.read_csv(uploaded_file)
                        
                        # Required columns check
                        required_cols = ['query_id', 'session_id', 'turn_id', 'ground_truth']
                        if not all(col in gt_df.columns for col in required_cols):
                            st.error(f" CSV must contain columns: {required_cols}")
                        else:
                            # Process and save to database immediately
                            evaluator = st.session_state.evaluator
                            updated_count = 0
                            failed_count = 0
                            
                            with st.spinner("Saving ground truth annotations to database..."):
                                for _, row in gt_df.iterrows():
                                    if pd.notna(row['ground_truth']) and str(row['ground_truth']).strip():
                                        success = evaluator.save_ground_truth_to_db(
                                            str(row['session_id']),
                                            int(row['turn_id']),
                                            str(row['ground_truth'])
                                        )
                                        
                                        if success:
                                            # Update evaluation data in memory
                                            for item in st.session_state.evaluation_data:
                                                if item['query_id'] == str(row['query_id']):
                                                    item['ground_truth'] = str(row['ground_truth'])
                                                    item['has_existing_gt'] = True
                                                    break
                                            updated_count += 1
                                        else:
                                            failed_count += 1
                            
                            if updated_count > 0:
                                st.success(f" Successfully saved {updated_count} ground truth annotations to database!")


                            
                            if failed_count > 0:
                                st.warning(f" Failed to save {failed_count} annotations")
                                
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            
            # Final ground truth summary
            final_gt_count = sum(1 for item in st.session_state.evaluation_data if item.get('ground_truth', '').strip())
            st.info(f" Final Status: {final_gt_count}/{total_queries} queries have ground truth")
    else:
        st.warning("Please extract evaluation data first.")

with tab4:
    st.header(" RAGAS Evaluation")
    
    if not st.session_state.models_configured:
        st.warning(" Please configure models first in the sidebar")
    elif st.session_state.evaluation_data and selected_metrics:
        
        # Check data requirements
        total_queries = len(st.session_state.evaluation_data)
        has_ground_truth = sum(1 for item in st.session_state.evaluation_data if item.get('ground_truth', '').strip())
        
        st.info(f" Dataset: {total_queries} queries | Ground truth: {has_ground_truth} queries")
        st.info(f" Using: {llm_model} + {embedding_model}")
        
        # Metric requirements check
        needs_gt_metrics = [m for m in selected_metrics if m in ["answer_correctness", "answer_similarity"]]
        can_run_metrics = [m for m in selected_metrics if m not in ["answer_correctness", "answer_similarity"]]
        
        if needs_gt_metrics and has_ground_truth == 0:
            st.warning(f" Metrics {needs_gt_metrics} require ground truth but none provided. Only {can_run_metrics} will be evaluated.")
            final_metrics = can_run_metrics
        else:
            final_metrics = selected_metrics
        
        if final_metrics:
            if st.button(" Run RAGAS Evaluation", type="primary"):
                with st.spinner("Running RAGAS evaluation..."):
                    try:
                        # Prepare dataset for RAGAS
                        eval_data = []
                        for item in st.session_state.evaluation_data:
                            ragas_item = {
                                "question": item["question"],
                                "answer": item["answer"],
                                "contexts": item["contexts"]
                            }
                            
                            # Add ground truth if available and needed
                            if item.get("ground_truth", "").strip() and any(m in final_metrics for m in ["answer_correctness", "answer_similarity"]):
                                ragas_item["ground_truth"] = item["ground_truth"]
                            
                            eval_data.append(ragas_item)
                        
                        # Create dataset
                        dataset = Dataset.from_list(eval_data)
                        
                        # Get metric objects with configured models
                        evaluator = st.session_state.evaluator
                        metrics = evaluator.get_metric_objects(final_metrics)
                        
                        # Run evaluation
                        raw_results = evaluate(dataset, metrics=metrics)
                        
                        # Parse results properly
                        scores, detailed_df = parse_ragas_results(raw_results)
                        
                        # Store results
                        st.session_state.ragas_results = {
                            "scores": scores,
                            "detailed_df": detailed_df,
                            "raw_results": raw_results,
                            "dataset_size": len(eval_data),
                            "metrics_used": final_metrics,
                            "models_used": {
                                "llm": llm_model,
                                "embeddings": embedding_model
                            },
                        }
                        
                        # Store RAGAS results back to database (ground truth already saved separately)
                        st.info(" Saving RAGAS scores to database...")
                        
                        # Prepare bulk update data for RAGAS scores only
                        bulk_update_data = []
                        
                        if detailed_df is not None and not detailed_df.empty:
                            # Use detailed results if available
                            for idx, (eval_item, result_row) in enumerate(zip(st.session_state.evaluation_data, detailed_df.iterrows())):
                                _, row_data = result_row
                                
                                # Extract individual scores for this query
                                individual_scores = {}
                                for metric in final_metrics:
                                    if metric in row_data and pd.notna(row_data[metric]):
                                        individual_scores[metric] = float(row_data[metric])
                                
                                bulk_update_data.append({
                                    "session_id": eval_item["session_id"],
                                    "turn_id": eval_item["turn_id"],
                                    "scores": individual_scores
                                    # Note: ground_truth already saved via annotation tab
                                })
                        else:
                            # Use average scores if detailed results not available
                            for eval_item in st.session_state.evaluation_data:
                                # Only include non-null scores
                                valid_scores = {k: v for k, v in scores.items() if v is not None and not pd.isna(v)}
                                
                                bulk_update_data.append({
                                    "session_id": eval_item["session_id"],
                                    "turn_id": eval_item["turn_id"],
                                    "scores": valid_scores
                                })
                        
                        # Update database with RAGAS scores only
                        session_logger = SessionLogger(chatdb_path)
                        updated_count = session_logger.bulk_update_ragas_scores(bulk_update_data)
                        
                        st.success(" RAGAS evaluation completed!")
                        st.success(f" Updated {updated_count} records in database with RAGAS scores")

                        
                    except Exception as e:
                        st.error(f"Error running RAGAS evaluation: {str(e)}")
                        st.exception(e)
        else:
            st.warning("No metrics selected for evaluation.")
    
    elif not st.session_state.evaluation_data:
        st.warning("Please extract evaluation data first.")
    elif not selected_metrics:
        st.warning("Please select at least one metric in the sidebar.")

with tab5:
    st.header(" RAGAS Results & Export")
    
    if st.session_state.ragas_results:
        results = st.session_state.ragas_results
        scores = results["scores"]
        detailed_df = results.get("detailed_df")
        
        st.subheader(" RAGAS Evaluation Results")
        st.markdown(f"**Models:** {results['models_used']['llm']} + {results['models_used']['embeddings']}")

        
        # Overall scores with color coding
        if scores:
            num_metrics = len(scores)
            cols = st.columns(min(4, num_metrics))
            
            for i, (metric, score) in enumerate(scores.items()):
                with cols[i % len(cols)]:
                    # Format metric name for display
                    display_name = metric.replace("_", " ").title()
                    
                    # Color coding based on score
                    if score >= 0.8:
                        score_color = "🟢"
                    elif score >= 0.6:
                        score_color = "🟡"
                    else:
                        score_color = "🔴"
                    
                    st.metric(
                        f"{score_color} {display_name}",
                        f"{score:.3f}",
                        help=f"RAGAS {metric} score using {results['models_used']['llm']}"
                    )
        else:
            st.warning("No scores extracted from RAGAS results")
        
        # Detailed results table
        st.subheader(" Detailed Scores")
        
        if detailed_df is not None and not detailed_df.empty:
            st.dataframe(detailed_df, use_container_width=True)
        
        # Export options
        st.subheader(" Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as JSON
            export_data = {
                "scores": scores,
                "metadata": {
                    "dataset_size": results['dataset_size'],
                    "metrics_used": results['metrics_used'],
                    "models_used": results['models_used'],
                }
            }
            results_json = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                " Download JSON",
                data=results_json,
                file_name=f"ragas_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export as CSV
            if detailed_df is not None and not detailed_df.empty:
                csv_data = detailed_df.to_csv(index=False)
                st.download_button(
                    " Download CSV",
                    data=csv_data,
                    file_name=f"ragas_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                # Export scores as CSV
                if scores:
                    scores_df = pd.DataFrame([scores])
                    csv_data = scores_df.to_csv(index=False)
                    st.download_button(
                        " Download CSV",
                        data=csv_data,
                        file_name=f"ragas_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            # Generate report
            report = f"""# RAGAS Evaluation Report


**Dataset Size:** {results['dataset_size']} queries  
**Metrics Evaluated:** {', '.join(results['metrics_used'])}  
**LLM Model:** {results['models_used']['llm']}  
**Embedding Model:** {results['models_used']['embeddings']}

## Overall Scores
"""
            for metric, score in scores.items():
                score_emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                report += f"- {score_emoji} **{metric.replace('_', ' ').title()}:** {score:.3f}\n"
            
            report += f"""
## Model Configuration
- **LLM:** {results['models_used']['llm']} for reasoning and evaluation
- **Embeddings:** {embedding_model} for semantic similarity

## Interpretation Guide
- **Faithfulness (>0.7):** Answer is well-grounded in retrieved context
- **Context Precision (>0.7):** Retrieved contexts are relevant to the question
- **Context Recall (>0.7):** All relevant information was retrieved
- **Answer Correctness (>0.7):** Factual accuracy compared to ground truth
- **Answer Similarity (>0.7):** Semantic similarity to ground truth

## Recommendations
"""
            # Add recommendations based on scores
            recommendations = []
            for metric, score in scores.items():
                if score < 0.7:
                    if metric == "faithfulness":
                        recommendations.append("- 🔴 Improve answer grounding - answers may hallucinate")
                    elif metric == "context_precision":
                        recommendations.append("- 🔴 Improve retrieval precision - too much irrelevant content")
                    elif metric == "context_recall":
                        recommendations.append("- 🔴 Improve retrieval recall - missing relevant information")
                    elif metric == "answer_correctness":
                        recommendations.append("- 🔴 Improve answer factual accuracy")
                    elif metric == "answer_similarity":
                        recommendations.append("- 🔴 Improve answer semantic quality")
            
            if not recommendations:
                report += "🟢 All metrics are performing well! No immediate issues detected.\n"
            else:
                report += "\n".join(recommendations) + "\n"
            
            st.download_button(
                 "Download Report",
                data=report,
                file_name=f"ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        # Summary insights
        st.subheader(" Insights & Recommendations")
        
        if scores:
            insights = []
            for metric, score in scores.items():
                if metric == "faithfulness" and score < 0.7:
                    insights.append("🔴 Low faithfulness - answers may not be grounded in retrieved context")
                elif metric == "context_precision" and score < 0.7:
                    insights.append("🔴 Low context precision - retrieval is bringing irrelevant information")
                elif metric == "context_recall" and score < 0.7:
                    insights.append("🔴 Low context recall - retrieval is missing relevant information")
                elif metric == "answer_correctness" and score < 0.7:
                    insights.append("🔴 Low answer correctness - answers may be factually incorrect")
                elif metric == "answer_similarity" and score < 0.7:
                    insights.append("🔴 Low answer similarity - answers differ semantically from ground truth")
                elif score >= 0.8:
                    insights.append(f"🟢 {metric.replace('_', ' ').title()} is performing excellently ({score:.3f})")
            
            if not any("🔴" in insight for insight in insights):
                st.success("🟢 All metrics look good! Your RAG system is performing well.")
            else:
                for insight in insights:
                    if "🔴" in insight:
                        st.warning(insight)
                    else:
                        st.success(insight)
        else:
            st.info("No insights available - check the raw results above.")
    
    else:
        st.info("Run RAGAS evaluation first to see results.")

# Footer
st.markdown("---")