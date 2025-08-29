import re
import traceback
import torch
import os
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image


# Load fine-tuned model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/LLaMA-Factory/saves_cot/Qwen2.5-VL-7B_lora_nocot",  # Modify to your fine-tuned model path
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/LLaMA-Factory/saves_cot/Qwen2.5-VL-7B_lora_nocot", padding_side='left', trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/LLaMA-Factory/saves_cot/Qwen2.5-VL-7B_lora_nocot", trust_remote_code=True)

def extract_mcq_only(question: str) -> str:
    # Match complete multiple choice question (based on (A)...(E)...)
    match = re.search(r"(Which of the following.*?\(E\)[^)]+?\.)", question)
    if match:
        return match.group(1).strip()
    return question

def load_data(input_file, image_dir):
    """Read input JSON, return image paths, questions and corresponding metadata lists."""
    data = []
    with open("./COT_0423/test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                line = json.loads(line)
                data.append(line)
    print(f"Read {len(data)} records")
    
    image_paths = []
    questions = []
    metadata = []

    for item in data:
        image_path = os.path.join(image_dir, item["image_path"])
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            continue
        image_paths.append(image_path)
        raw_question = item["conversation"]["Question"]
        questions.append(extract_mcq_only(raw_question))
        metadata.append({
            "image_path": item["image_path"],
            "coordinates": item["coordinates"],
            "type": item["type"],
            "answer": item["conversation"]["Answer"],
            # Adding question field for later reference
            "question": extract_mcq_only(raw_question)
        })
    
    return image_paths, questions, metadata


def batch_predict(batch_image_paths, batch_questions, model, processor, tokenizer):
    """Predict for a batch of images and questions, return list of model's raw outputs."""
    results = []
    batch_size = len(batch_image_paths)
    
    # Load images
    images = [Image.open(path).convert("RGB") for path in batch_image_paths]
    
    # Construct message list for Qwen2.5-VL format
    batch_messages = []
    for image, question in zip(images, batch_questions):
        messages = [
            {
                "role": "system",
                "content": "You are an AI model for anomaly detection. The format of the anomaly region coordinates should be <boxes>[[(?,?),(?,?)],...] </boxes>, that is, the abnormal area may be multiple or even one."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": question + " Please present your answer with: The answer is (X). The anomaly region coordinates are [[(?, ?), (?, ?)],......]."
                    }
                ]
            }
        ]
        batch_messages.append(messages)
    
    try:
        # Batch processing
        batch_texts = []
        batch_images = []
        batch_videos = []
        
        # Process each sample
        for messages in batch_messages:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            batch_texts.append(text)
            if image_inputs:
                batch_images.append(image_inputs[0])
            if video_inputs:
                batch_videos.extend(video_inputs)
        
        # Process inputs in batch
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            videos=batch_videos if batch_videos else None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate model output
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id
        )
        
        # Decode output for each sample
        responses = []
        for j, item_ids in enumerate(generated_ids):
            item_input_length = inputs.input_ids[j].shape[0]
            generated_ids_trimmed = item_ids[item_input_length:]
            response = processor.decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            responses.append(response)
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        traceback.print_exc()
        responses = ["Error"] * len(batch_image_paths)
    
    return responses


def load_processed_image_paths(output_file):
    """Read existing output file, return set of processed image_paths.
       Uses JSON Lines format, one JSON object per line."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed.add(record["image_path"])
                    except Exception as e:
                        print(f"Failed to read a line: {e}")
    return processed


def main(batch_size = 8):
    input_file = "/root/autodl-tmp/LLaMA-Factory/COT_0423/test_without_reasoning.jsonl"  # Modify to your input file path
    output_file = "/root/autodl-tmp/LLaMA-Factory/AnomalyCoT_output/Qwen2.5-VL-7B_lora_nocot.jsonl"  # Modify to your fine-tuned model's output file path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    image_dir = "./COT_0423"  # Modify to your image directory

    image_paths, questions, metadata = load_data(input_file, image_dir)
    processed_image_paths = load_processed_image_paths(output_file)

    with open(output_file, "a", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch progress"):
            batch_image_paths = image_paths[i:i+batch_size]
            batch_questions = questions[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]

            # Skip already processed images
            unprocessed_indices = [
                idx for idx, meta in enumerate(batch_metadata)
                if meta["image_path"] not in processed_image_paths
            ]
            if not unprocessed_indices:
                continue

            # Filter unprocessed samples
            batch_image_paths = [batch_image_paths[idx] for idx in unprocessed_indices]
            batch_questions   = [batch_questions[idx]   for idx in unprocessed_indices]
            batch_metadata    = [batch_metadata[idx]    for idx in unprocessed_indices]

            responses = batch_predict(batch_image_paths, batch_questions, model, processor, tokenizer)

            for meta, response in zip(batch_metadata, responses):
                result_item = {
                    "image_path": meta["image_path"],
                    "coordinates": meta["coordinates"],
                    "type": meta["type"],
                    "ground_truth_answer": meta["answer"],
                    "qwen_output": response,
                    "question": meta["question"]
                }
                
                # Write to file immediately to avoid data loss in case of interruption
                out_f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                out_f.flush()
                
                # Add to processed set
                processed_image_paths.add(meta["image_path"])


if __name__ == "__main__":
    main(batch_size=8) 