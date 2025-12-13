import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import re
import random
import json
import argparse
import glob
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import torch
from torch.utils.data import Dataset, DataLoader

# Transformers / PEFT - Modified for T5
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel

# ==========================================
# 1. DATA PROCESSING FOR PHASE 1
# ==========================================

class Phase1DataProcessor:
    @staticmethod
    def scramble_word(word):
        if not isinstance(word, str): 
            return ""
        clean = re.sub(r'[^a-zA-Z]', '', word)
        if len(clean) < 2:
            return clean
        chars = list(clean)
        random.shuffle(chars)
        return "".join(chars)

    @staticmethod
    def load_isarcasm(path):
        """Load and process iSarcasm dataset"""
        print(f"Loading iSarcasm data from {path}...")
        try:
            df = pd.read_csv(path)
            cols = df.columns.str.lower()
            
            # Handle different column names
            if 'tweet' in cols: 
                df = df.rename(columns={'tweet': 'input'})
            elif 'text' in cols: 
                df = df.rename(columns={'text': 'input'})
            
            label_col = 'sarcastic' if 'sarcastic' in cols else 'label'
            
            processed_data = []
            for _, row in df.iterrows():
                text = str(row['input'])
                label = row[label_col]
                
                # Normalize labels
                if str(label) in ['1', '1.0', 'sarcastic', 'true']:
                    target = "sarcastic"
                else:
                    target = "not sarcastic"
                
                processed_data.append({
                    "input": text,
                    "target": target,
                    "task_type": "sarcasm"
                })
            
            return pd.DataFrame(processed_data)
        except Exception as e:
            print(f"Error loading iSarcasm: {e}")
            return pd.DataFrame()

    @staticmethod
    def prepare_phase1_datasets(train_path, isarcasm_path, test_size=0.2):
        """
        Prepare Phase 1 datasets with train/test split
        Returns: train_df, test_df
        """
        print(f"\n=== PREPARING PHASE 1 DATASETS ===")
        
        # Load Guardian data for anagram generation
        print(f"Loading Guardian data from {train_path}...")
        try:
            guardian_df = pd.read_csv(train_path, dtype=str).dropna()
        except FileNotFoundError:
            print(f"WARNING: File {train_path} not found. Creating dummy data for testing code structure.")
            guardian_df = pd.DataFrame({'target': ['hello', 'world', 'python']})

        # Create anagram/unscramble dataset
        anagram_data = []
        if 'target' in guardian_df.columns:
            for tgt in guardian_df['target']:
                tgt = str(tgt).strip()
                if len(tgt) > 2 and tgt.isalpha():  # Only use valid alphabetic words
                    scrambled = Phase1DataProcessor.scramble_word(tgt)
                    if scrambled != tgt:  # Ensure it's actually scrambled
                        anagram_data.append({
                            "input": scrambled,
                            "target": tgt.lower(),
                            "task_type": "unscramble"
                        })
        
        anagram_df = pd.DataFrame(anagram_data)
        print(f"Created {len(anagram_df)} anagram samples")
        
        # Load sarcasm dataset
        sarcasm_df = Phase1DataProcessor.load_isarcasm(isarcasm_path)
        print(f"Loaded {len(sarcasm_df)} sarcasm samples")
        
        if len(anagram_df) == 0 and len(sarcasm_df) == 0:
            print("ERROR: No data loaded. Exiting.")
            return pd.DataFrame(), pd.DataFrame()

        # Split each task separately to maintain balance
        if len(anagram_df) > 0:
            anagram_train = anagram_df.sample(frac=1-test_size, random_state=42)
            anagram_test = anagram_df.drop(anagram_train.index)
        else:
            anagram_train, anagram_test = pd.DataFrame(), pd.DataFrame()
        
        if len(sarcasm_df) > 0:
            sarcasm_train = sarcasm_df.sample(frac=1-test_size, random_state=42)
            sarcasm_test = sarcasm_df.drop(sarcasm_train.index)
        else:
            sarcasm_train, sarcasm_test = pd.DataFrame(), pd.DataFrame()
        
        # Combine and shuffle
        train_df = pd.concat([anagram_train, sarcasm_train]).sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = pd.concat([anagram_test, sarcasm_test]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nTrain set: {len(train_df)} samples")
        print(f"  - Anagrams: {len(anagram_train)}")
        print(f"  - Sarcasm: {len(sarcasm_train)}")
        print(f"\nTest set: {len(test_df)} samples")
        
        return train_df, test_df


class Phase1Dataset(Dataset):
    """
    Dataset for Phase 1 tasks: unscramble and sarcasm detection
    """
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def build_prompt(self, inp_text, task):
        """Build task-specific prompts"""
        if task == "unscramble":
            return f"unscramble: {inp_text}"
        elif task == "sarcasm":
            return f"classify sarcasm: {inp_text}"
        else:
            return inp_text

    def __getitem__(self, index):
        row = self.data.iloc[index]
        inp_text = str(row['input']).strip()
        target_text = str(row['target']).strip()
        task = row['task_type']

        # Build prompt
        prompt = self.build_prompt(inp_text, task)

        # Tokenize input (encoder)
        input_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize target (decoder)
        target_enc = self.tokenizer(
            target_text,
            truncation=True,
            max_length=64,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = input_enc.input_ids.squeeze(0)
        attention_mask = input_enc.attention_mask.squeeze(0)
        labels = target_enc.input_ids.squeeze(0)
        
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'raw_input': inp_text,
            'target': target_text,
            'task_type': task
        }


# ==========================================
# 2. T5 + LoRA MODEL WRAPPER
# ==========================================

class Phase1T5Model:
    def __init__(self, model_name='t5-base', checkpoint_path=None, lora_r=8, lora_alpha=16, lora_dropout=0.05, device=None):
        """
        Args:
            model_name: Base HuggingFace model
            checkpoint_path: Path to a saved PEFT adapter to load (for resuming training)
            lora_r, lora_alpha, lora_dropout: LoRA config params (used only if checkpoint_path is None)
        """
        # Device selection
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"\nInitializing {model_name} on {self.device}...")

        # Load tokenizer & Base model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔄 Loading existing LoRA adapter from {checkpoint_path}...")
            # Load the PEFT model with is_trainable=True to allow continued training
            self.model = PeftModel.from_pretrained(
                self.model, 
                checkpoint_path, 
                is_trainable=True
            )
        else:
            print("🆕 Initializing new LoRA adapters...")
            # Configure LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q", "v"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model.to(self.device)
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """Helper to print number of trainable params"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

    def generate(self, prompt, max_new_tokens=32, num_beams=1):
        """Generate response for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================

def train_epoch(model_wrapper, dataloader, optimizer, epoch_num):
    """Train for one epoch"""
    model = model_wrapper.model
    model.train()
    
    total_loss = 0.0
    task_losses = defaultdict(list)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num}")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(model_wrapper.device)
        attention_mask = batch['attention_mask'].to(model_wrapper.device)
        labels = batch['labels'].to(model_wrapper.device)
        task_types = batch['task_type']

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Track losses per task
        for i, task in enumerate(task_types):
            # Approximate task tracking (since loss is averaged over batch)
            task_losses[task].append(loss.item())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    
    # Print task-specific losses
    print(f"\nEpoch {epoch_num} Summary:")
    print(f"  Overall Loss: {avg_loss:.4f}")
    
    return avg_loss


def evaluate_phase1(model_wrapper, test_df, batch_size=8):
    """
    Comprehensive evaluation for Phase 1 tasks
    """
    print("\n" + "="*60)
    print("PHASE 1 EVALUATION")
    print("="*60)
    
    model_wrapper.model.eval()
    
    # Separate by task
    anagram_data = test_df[test_df['task_type'] == 'unscramble']
    sarcasm_data = test_df[test_df['task_type'] == 'sarcasm']
    
    # ===== ANAGRAM/UNSCRAMBLE EVALUATION =====
    anagram_accuracy = 0
    anagram_results = []
    
    if len(anagram_data) > 0:
        print("\n--- ANAGRAM UNSCRAMBLING TASK ---")
        anagram_correct = 0
        print(f"\nEvaluating {len(anagram_data)} anagram samples...")
        for idx, row in tqdm(anagram_data.iterrows(), total=len(anagram_data), desc="Anagrams"):
            inp = str(row['input'])
            target = str(row['target']).strip().lower()
            
            prompt = f"unscramble: {inp}"
            prediction = model_wrapper.generate(prompt, max_new_tokens=32).lower()
            
            is_correct = (prediction == target)
            if is_correct:
                anagram_correct += 1
            
            anagram_results.append({
                "input": inp,
                "target": target,
                "prediction": prediction,
                "correct": is_correct
            })
        
        anagram_accuracy = (anagram_correct / len(anagram_data)) * 100
        print(f"\nAnagram Results: Accuracy: {anagram_accuracy:.2f}%")
    
    # ===== SARCASM DETECTION EVALUATION =====
    sarcasm_accuracy = 0
    sarcasm_results = []
    precision, recall, f1 = 0, 0, 0
    
    if len(sarcasm_data) > 0:
        print("\n--- SARCASM DETECTION TASK ---")
        sarcasm_predictions = []
        sarcasm_targets = []
        
        print(f"\nEvaluating {len(sarcasm_data)} sarcasm samples...")
        for idx, row in tqdm(sarcasm_data.iterrows(), total=len(sarcasm_data), desc="Sarcasm"):
            inp = str(row['input'])
            target = str(row['target']).strip().lower()
            
            prompt = f"classify sarcasm: {inp}"
            prediction = model_wrapper.generate(prompt, max_new_tokens=16).lower()
            
            # Normalize predictions
            if "sarcastic" in prediction and "not" not in prediction:
                pred_label = "sarcastic"
            else:
                pred_label = "not sarcastic"
            
            sarcasm_predictions.append(pred_label)
            sarcasm_targets.append(target)
            
            sarcasm_results.append({
                "input": inp[:50],
                "target": target,
                "prediction": pred_label,
                "correct": pred_label == target
            })
        
        # Compute metrics
        sarcasm_accuracy = accuracy_score(sarcasm_targets, sarcasm_predictions) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            sarcasm_targets, 
            sarcasm_predictions, 
            average='weighted',
            zero_division=0
        )
        print(f"\nSarcasm Results: Accuracy: {sarcasm_accuracy:.2f}% | F1: {f1:.4f}")

    # ===== OVERALL SUMMARY =====
    total_correct = sum(1 for r in anagram_results if r['correct']) + sum(1 for r in sarcasm_results if r['correct'])
    total_samples = len(test_df)
    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
    results_df = pd.DataFrame({
        'Task': ['Anagram', 'Sarcasm', 'Overall'],
        'Accuracy': [anagram_accuracy, sarcasm_accuracy, overall_accuracy]
    })
    
    all_results = anagram_results + sarcasm_results
    detailed_df = pd.DataFrame(all_results)
    
    return {
        'results_summary': results_df,
        'detailed_results': detailed_df
    }


# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="T5 LoRA Training & Resume Script")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a saved checkpoint to resume from (e.g., 'checkpoints/epoch-3')")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch to start training from (useful if resuming)")
    parser.add_argument("--epochs", type=int, default=6, help="Total number of epochs to train")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # File paths
    BASE_DIR = './LLM_proj/llm_proj_data'
    TRAIN_FILE = os.path.join(BASE_DIR, 'guardian/train.csv')
    ISARCASM_FILE = os.path.join(BASE_DIR, 'isarcasm/isarcasm_train.csv')
    CHECKPOINT_DIR = "./checkpoints"

    # Prepare datasets
    train_df, test_df = Phase1DataProcessor.prepare_phase1_datasets(
        TRAIN_FILE, 
        ISARCASM_FILE, 
        test_size=0.2
    )
    
    if len(train_df) == 0:
        return

    # Initialize model (Load checkpoint if provided)
    model_name = 't5-base'
    model = Phase1T5Model(
        model_name=model_name, 
        checkpoint_path=args.checkpoint,
        lora_r=8, 
        lora_alpha=16, 
        lora_dropout=0.05
    )

    # Hyperparameters
    BATCH_SIZE = 16
    MAX_LEN = 128
    LEARNING_RATE = 1e-3

    # Create dataloaders
    train_dataset = Phase1Dataset(train_df, model.tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.model.parameters()), 
        lr=LEARNING_RATE
    )
    
    # Attempt to load optimizer state if resuming (optional but recommended for perfect resume)
    if args.checkpoint:
        opt_path = os.path.join(args.checkpoint, "optimizer.pt")
        if os.path.exists(opt_path):
            print(f"🔄 Loading optimizer state from {opt_path}")
            try:
                optimizer.load_state_dict(torch.load(opt_path))
            except Exception as e:
                print(f"⚠️ Could not load optimizer state (might be different mismatch): {e}")

    # Training Loop
    print("\n" + "="*60)
    print("PHASE 1 TRAINING: FOUNDATION SKILLS")
    print("="*60)
    print(f"Resume Checkpoint: {args.checkpoint if args.checkpoint else 'None'}")
    print(f"Start Epoch: {args.start_epoch}")
    print(f"Total Epochs: {args.epochs}")
    
    if args.start_epoch > args.epochs:
        print("Start epoch is greater than total epochs. Skipping training phase.")
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        
        # Save Checkpoint per Epoch
        epoch_save_path = os.path.join(CHECKPOINT_DIR, f"epoch-{epoch}")
        print(f"💾 Saving checkpoint to {epoch_save_path}...")
        model.model.save_pretrained(epoch_save_path)
        model.tokenizer.save_pretrained(epoch_save_path)
        
        # Save optimizer state separately
        torch.save(optimizer.state_dict(), os.path.join(epoch_save_path, "optimizer.pt"))

    # Final Save
    final_save_dir = "./phase1_t5_lora_final"
    model.model.save_pretrained(final_save_dir)
    print(f"✅ Final model saved to {final_save_dir}")

    # Evaluation
    metrics = evaluate_phase1(model, test_df, batch_size=BATCH_SIZE)

    # Save results
    metrics['results_summary'].to_csv("phase1_summary.csv", index=False)
    metrics['detailed_results'].to_csv("phase1_detailed_results.csv", index=False)
    
    print(f"\n📊 Results saved.")

if __name__ == "__main__":
    main()
