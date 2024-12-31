from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import json
import os
import random

def prepare_training_data(output_path="training_data.txt", max_examples=100000):
    print("Downloading and preparing datasets...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Loading Natural Questions dataset...")
    nq_dataset = load_dataset("natural_questions", split="train", streaming=True)
    print("Loading SQuAD dataset...")
    squad_dataset = load_dataset("squad_v2", split="train", streaming=True)
    print("Processing and combining datasets...")
    print("i love print statements bhaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        qa_count = 0
        for item in tqdm(nq_dataset.take(max_examples), desc="Processing Natural Questions", total=max_examples):
            try:
                if 'question' in item and 'annotations' in item:
                    question = item['question']['text']
                    answer = ""
                    if item.get('annotations') and isinstance(item['annotations'], list):
                        for annotation in item['annotations']:
                            if annotation.get('short_answers') and len(annotation['short_answers']) > 0:
                                answer = annotation['short_answers'][0]['text']
                                break
                    
                    if question and answer and len(question.split()) >= 3 and len(str(answer).split()) >= 1:
                        
                        f.write(f"Question: {question}\nAnswer: {answer}\n\n")
                        qa_count += 1
            except Exception as e:
                continue  
        
        
        squad_count = 0
        for item in tqdm(squad_dataset.take(max_examples), desc="Processing SQuAD", total=max_examples):
            try:
                question = item['question']
                answers = item.get('answers', {})
                answer = answers.get('text', [''])[0] if answers else ""
                
                if question and answer and len(question.split()) >= 3 and len(answer.split()) >= 1:
                    f.write(f"Question: {question}\nAnswer: {answer}\n\n")
                    squad_count += 1
            except Exception as e:
                continue 
    
    print(f"\nData preparation completed!")
    print(f"Processed {qa_count} Natural Questions examples")
    print(f"Processed {squad_count} SQuAD examples")
    print(f"Combined dataset saved to: {output_path}")
    
    return output_path

def get_dataset_stats(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    qa_pairs = content.count('Question:')
    
    
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)  # MB
    size_gb = size_mb / 1024  # GB
    
    
    questions = [q.strip() for q in content.split('Question:')[1:]]
    avg_q_length = sum(len(q.split('\n')[0].split()) for q in questions) / len(questions)
    
    print("\nDetailed Dataset Statistics:")
    print(f"Number of Q&A pairs: {qa_pairs:,}")
    print(f"Average question length: {avg_q_length:.1f} words")
    print(f"File size: {size_bytes:,} bytes")
    print(f"         = {size_mb:.2f} MB")
    print(f"         = {size_gb:.3f} GB")

def validate_data_quality(file_path, sample_size=10):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    
    qa_pairs = content.split('\n\n')
    samples = random.sample(qa_pairs, min(sample_size, len(qa_pairs)))
    
    print("\nRandom Sample of Q&A Pairs:")
    print("="*50)
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(sample)
        print("-"*50)

if __name__ == "__main__":
    output_path = "Deep_learning/Transformer/data/training_data.txt"
    data_path = prepare_training_data(output_path, max_examples=100000)
    get_dataset_stats(data_path)
    validate_data_quality(data_path)