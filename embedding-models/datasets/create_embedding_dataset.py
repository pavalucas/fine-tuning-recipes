import json
import os
import random
import logging
import sys
from typing import List, Dict, Tuple
import pandas as pd
from aixplain.factories.model_factory import ModelFactory
from aixplain.modules.model.llm_model import LLM

# Configure logging to ensure immediate output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True  # Force reconfiguration of root logger
)

def load_case(file_path: str) -> Dict:
    """Load a case from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_case_text(case: Dict) -> str:
    """Extract the full text of a case."""
    # Combine title, judgment text, and appeal text
    text_parts = []
    
    if "عنوان الحكم" in case:
        text_parts.append(case["عنوان الحكم"])
    
    if "نص الحكم" in case:
        text_parts.extend(case["نص الحكم"])
    
    if "الاستئناف" in case:
        text_parts.extend(case["الاستئناف"])
    
    return "\n".join(text_parts)

def get_case_number(case: Dict) -> str:
    """Extract the case number from the case data."""
    if "بيانات الحكم" in case:
        case_data = case["بيانات الحكم"]
        # Try to get the case number from the title if available
        if "عنوان الحكم" in case:
            title = case["عنوان الحكم"]
            # Look for pattern like "رقم XX لعام"
            import re
            match = re.search(r'رقم\s+(\d+)\s+لعام', title)
            if match:
                return match.group(1)
        
        # Fallback to using the filename as case number
        return os.path.splitext(os.path.basename(case.get("_file_path", "")))[0]
    
    return "unknown"

def generate_queries(case: Dict, llm: LLM, chunk: str = None) -> List[str]:
    """Generate potential queries for a case using an LLM."""
    logging.info(f"Generating queries for case {get_case_number(case)}")
    
    # Prepare the case information for the LLM
    case_info = {
        "title": case.get("عنوان الحكم", ""),
        "categories": case.get("التصنيف", []),
        "court": case.get("بيانات الحكم", {}).get(" المحكمة العامة", ""),
        "case_number": get_case_number(case),
        "summary": extract_case_text(case)[:1000]  # First 1000 chars as context
    }
    
    # Create the messages for the LLM
    messages = [
        {
            "role": "system",
            "content": "You are a legal expert creating search queries for a legal document retrieval system."
        },
        {
            "role": "user",
            "content": f"""
            Given the following case information and document chunk, generate 3 different natural language queries in Arabic that someone might use to search for this specific part of the case. The query must be general so DO NOT use words like "this case" or "that case". You MUST use the name of the companies and the parties involved in the case, if they are explicitly mentioned. DO NOT use the case number or dates in the queries.
            Make the queries diverse and natural, focusing on the content of the provided chunk.
            
            Case Information:
            Title: {case_info['title']}
            Categories: {', '.join(case_info['categories'])}
            Court: {case_info['court']}
            Case Number: {case_info['case_number']}
            
            Document Chunk:
            {chunk if chunk else case_info['summary']}
            
            Generate 3 different queries in Arabic. Each query should be on a new line.
            """
        }
    ]
    
    try:
        # Get response from LLM using call method
        response = llm.run(history=messages)
        
        # Split the response into individual queries
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        
        # If we got fewer than 3 queries, generate more
        while len(queries) < 3:
            additional_response = llm.call(messages)
            additional_queries = [q.strip() for q in additional_response.split('\n') if q.strip()]
            queries.extend(additional_queries)
        
        logging.info(f"Successfully generated {len(queries)} queries for case {get_case_number(case)}")
        return queries[:3]  # Return exactly 3 queries
    except Exception as e:
        logging.error(f"Error generating queries for case {get_case_number(case)}: {str(e)}")
        return ["Error generating queries"] * 3  # Return placeholder queries

def chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split a document into overlapping chunks efficiently."""
    logging.info(f"Starting document chunking with text length: {len(text)}")
    
    if not text:
        logging.warning("Empty text provided for chunking")
        return []
    
    chunks = []
    text_length = len(text)
    
    # Calculate the step size (chunk_size - overlap)
    step = chunk_size - overlap
    if step <= 0:
        logging.error(f"Invalid chunk configuration: chunk_size ({chunk_size}) must be greater than overlap ({overlap})")
        return [text]  # Return the whole text as one chunk
    
    # Process the text in steps
    for start in range(0, text_length, step):
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        
        # If we've reached the end, break
        if end == text_length:
            break
    
    logging.info(f"Created {len(chunks)} chunks with average size: {sum(len(c) for c in chunks)/len(chunks):.2f} characters")
    return chunks

def process_cases_in_batches(cases: List[Dict], batch_size: int = 2) -> List[Dict]:
    """Process cases in batches to manage memory usage."""
    logging.info("Starting process_cases_in_batches")
    triplets = []
    total_batches = (len(cases) + batch_size - 1) // batch_size
    logging.info(f"Total number of batches to process: {total_batches}")
    
    for batch_num in range(total_batches):
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting batch {batch_num + 1}/{total_batches}")
        
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(cases))
        batch_cases = cases[start_idx:end_idx]
        
        logging.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_cases)} cases)")
        logging.info(f"Case indices: {start_idx} to {end_idx}")
        
        # Create a mapping of case ID to chunks for this batch
        logging.info("Starting case chunking process")
        case_chunks = {}
        for i, case in enumerate(batch_cases):
            logging.info(f"Processing case {i+1}/{len(batch_cases)} in current batch")
            case_id = get_case_number(case)
            if case_id != "unknown":
                logging.info(f"Extracting text for case {case_id}")
                text = extract_case_text(case)
                logging.info(f"Chunking text for case {case_id}")
                chunks = chunk_document(text)
                case_chunks[case_id] = chunks
                logging.info(f"Processed case {case_id} with {len(chunks)} chunks")
            else:
                logging.warning(f"Skipping case with unknown ID")
        
        if case_chunks:
            logging.info(f"Starting triplet generation for batch {batch_num + 1}")
            # Generate triplets for this batch
            batch_triplets = create_triplets_for_batch(batch_cases, case_chunks)
            triplets.extend(batch_triplets)
            logging.info(f"Generated {len(batch_triplets)} triplets in batch {batch_num + 1}")
            
            # Save progress after each batch
            if triplets:
                logging.info("Saving current progress to CSV")
                df = pd.DataFrame(triplets)
                df.to_csv("embedding_training_dataset.csv", index=False, encoding='utf-8')
                logging.info(f"Saved {len(triplets)} triplets to CSV")
        else:
            logging.warning(f"No valid case chunks found in batch {batch_num + 1}")
        
        logging.info(f"Completed batch {batch_num + 1}/{total_batches}")
        logging.info(f"{'='*50}\n")
    
    logging.info("Completed all batches in process_cases_in_batches")
    return triplets

def create_triplets_for_batch(cases: List[Dict], case_chunks: Dict[str, List[str]], num_triplets: int = 2) -> List[Dict]:
    """Create triplets for a batch of cases."""
    triplets = []
    llm = ModelFactory.get("6646261c6eb563165658bbb1")  # Create a new LLM instance for this batch
    
    for i in range(num_triplets):
        logging.info(f"Creating triplet {i+1}/{num_triplets} for current batch")
        try:
            # Select a random case as the positive example
            positive_case_id = random.choice(list(case_chunks.keys()))
            positive_case = next(c for c in cases if get_case_number(c) == positive_case_id)
            positive_chunks = case_chunks[positive_case_id]
            
            # Select a random chunk from the positive case
            positive_chunk = random.choice(positive_chunks)
            logging.info(f"Selected chunk from case {positive_case_id} with length {len(positive_chunk)}")
            
            # Generate queries specifically about this chunk
            logging.info("Generating queries for selected chunk")
            queries = generate_queries(positive_case, llm, positive_chunk)
            query = random.choice(queries)
            
            # Select a random different case and chunk as the negative example
            negative_case_id = random.choice([cid for cid in case_chunks.keys() if cid != positive_case_id])
            negative_chunks = case_chunks[negative_case_id]
            negative_chunk = random.choice(negative_chunks)
            
            triplets.append({
                "query": query,
                "positive_doc": positive_chunk,
                "negative_doc": negative_chunk,
                "positive_case_id": positive_case_id,
                "negative_case_id": negative_case_id
            })
            logging.info(f"Successfully created triplet {i+1}")
        except Exception as e:
            logging.error(f"Error creating triplet {i+1}: {str(e)}")
    
    return triplets

def main():
    logging.info("Starting embedding dataset creation")
    
    # Load all cases from the sample_data directory
    sample_data_dir = "sample_data"
    cases = []
    logging.info(f"Loading cases from {sample_data_dir}")
    
    for filename in os.listdir(sample_data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(sample_data_dir, filename)
            try:
                case = load_case(file_path)
                case["_file_path"] = file_path
                cases.append(case)
                logging.info(f"Loaded case {filename}")
            except Exception as e:
                logging.error(f"Error loading case {filename}: {str(e)}")
    
    logging.info(f"Successfully loaded {len(cases)} cases")
    
    # Process cases in batches
    triplets = process_cases_in_batches(cases)
    
    # Print sample triplets
    logging.info("Sample triplets created:")
    for i, triplet in enumerate(triplets[:3]):
        print(f"\nTriplet {i+1}:")
        print(f"Query: {triplet['query']}")
        print(f"Positive Chunk: {triplet['positive_doc'][:200]}...")
        print(f"Negative Chunk: {triplet['negative_doc'][:200]}...")
        print(f"Positive Case ID: {triplet['positive_case_id']}")
        print(f"Negative Case ID: {triplet['negative_case_id']}")
        print("-" * 80)
    
    logging.info("Embedding dataset creation completed successfully")

if __name__ == "__main__":
    main() 