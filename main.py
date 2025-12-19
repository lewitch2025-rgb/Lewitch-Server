import os
import torch
import uvicorn
import glob
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles  # <--- NEW IMPORT
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from datasets import Dataset

# --- INITIALIZE APP ---
app = FastAPI(title="Lewitch AI Server", version="1.0")

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
MODEL_ID = "huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated"
DATASET_FOLDER = "datasets"
OUTPUT_DIR = "output"
ADAPTER_NAME = "lewitch_adapter"

# Global variables
model = None
tokenizer = None

# --- PYDANTIC MODELS ---
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

class TrainRequest(BaseModel):
    epochs: int = 1
    batch_size: int = 2

# --- LIFECYCLE MANAGEMENT ---
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    print(f"Loading Model: {MODEL_ID}...")
    
    # 4-Bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load local adapters if available
        adapter_path = os.path.join(OUTPUT_DIR, ADAPTER_NAME)
        if os.path.exists(adapter_path):
            print("Found trained adapters! Loading custom Lewitch knowledge...")
            model = PeftModel.from_pretrained(model, adapter_path)
        
        print("System Online: DeepSeek-R1 Abliterated is ready.")
    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODEL: {e}")

# --- HELPER: TRAINING FUNCTION ---
def run_training_task(epochs: int, batch_size: int):
    global model, tokenizer
    print("--- STARTING TRAINING SEQUENCE ---")
    
    txt_files = glob.glob(os.path.join(DATASET_FOLDER, "*.txt"))
    if not txt_files:
        print("No text files found in datasets/ folder.")
        return

    raw_text = []
    for file in txt_files:
        with open(file, "r", encoding="utf-8") as f:
            raw_text.append(f.read())
    
    dataset = Dataset.from_dict({"text": raw_text})
    print(f"Training on {len(txt_files)} files...")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=epochs,
        save_strategy="no",
        fp16=True,
        optim="paged_adamw_8bit"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.train()
    
    save_path = os.path.join(OUTPUT_DIR, ADAPTER_NAME)
    trainer.model.save_pretrained(save_path)
    print(f"Training Complete. Adapters saved to {save_path}")
    print("Please restart the server to load the new knowledge fully.")

# --- API ENDPOINTS ---

@app.get("/api/status")
async def status_check():
    return {"status": "online", "model": MODEL_ID}

@app.post("/chat")
async def generate_chat(request: ChatRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = generated_text.replace(request.prompt, "").strip()

    return {"response": response_text, "raw": generated_text}

@app.post("/train")
async def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    background_tasks.add_task(run_training_task, request.epochs, request.batch_size)
    return {"status": "Training initiated", "info": "Check terminal logs"}

# --- MOUNT WEBSITE ---
# This serves the 'src' folder at the root URL '/'
# html=True means it will look for 'index.html' automatically
app.mount("/", StaticFiles(directory="src", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
