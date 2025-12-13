# neurocrypt_t5_lora_with_decomposer.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import re
import random
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# Transformers / PEFT - Modified for T5
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

# NLP libs (spaCy + NLTK WordNet)
try:
    import spacy
except Exception:
    raise ImportError("spaCy not installed. Run `pip install spacy` and `python -m spacy download en_core_web_sm`")

import nltk
from nltk.corpus import wordnet as wn

# Ensure NLTK wordnet is available
try:
    wn.ensure_loaded()
except Exception:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# ------------------------
# Decomposer + Linguistic KG
# ------------------------

class ClueDecomposer:
    """
    Heuristic decomposer for cryptic crossword clues.
    Produces a dict with:
      - clue_text
      - length (int or list)
      - tokens, pos_tags
      - predicted_type (anagram/container/hidden/reversal/charade/double/definition)
      - indicator_tokens (list)
      - fodder_spans (list of token spans likely to be used in wordplay)
      - wordnet_hints {token: [lemmas/hypernyms]}
    """

    # indicator lexica (non-exhaustive; expand as needed)
    ANAGRAM_INDICATORS = {"agitated","drunk","shaken","mixed","stirred","messed","wild","scrambled","confused","roiled","upset"}
    HIDDEN_INDICATORS = {"in","inside","within","hides","hidden","concealed","contained","somewhere"}
    CONTAINER_INDICATORS = {"around","surrounding","about","embracing","holding","wraps","wrapped"}
    REVERSAL_INDICATORS = {"back","returned","reversed","going back","about"}
    CHARAD_INDICATORS = {"and","with","plus"}
    DEFINITION_SEPARATORS = {":", "—", "-", "–"}

    def __init__(self, spacy_model="en_core_web_sm", use_conceptnet=False):
        self.nlp = spacy.load(spacy_model)
        self.use_conceptnet = use_conceptnet
        # ConceptNet access is optional and disabled by default (internet required)
        if use_conceptnet:
            try:
                import requests
                self.requests = requests
            except Exception:
                self.use_conceptnet = False

    @staticmethod
    def extract_length(clue: str):
        """
        Extract length hint like (5) or (3,4) returns tuple or list
        """
        match = re.search(r'\(([\d,\- ]+)\)\s*$', clue.strip())
        if not match:
            # sometimes length is within clue text; try to find e.g. (3,4) anywhere
            match = re.search(r'\(([\d,\- ]+)\)', clue)
        if not match:
            return None
        raw = match.group(1)
        # handle ranges e.g. 3-4 -> choose list [3,4]
        parts = [p.strip() for p in re.split(r'[,/]', raw) if p.strip()]
        lengths = []
        for p in parts:
            if '-' in p:
                a,b = p.split('-',1)
                lengths.extend(list(range(int(a), int(b)+1)))
            else:
                try:
                    lengths.append(int(p))
                except:
                    pass
        return sorted(set(lengths))

    def pos_and_tokens(self, clue: str):
        doc = self.nlp(clue)
        tokens = [t.text for t in doc]
        pos = [t.pos_ for t in doc]
        lemmas = [t.lemma_ for t in doc]
        return tokens, pos, lemmas, doc

    def detect_indicators(self, tokens_lower: List[str]) -> Dict[str, List[int]]:
        found = defaultdict(list)
        for i,tk in enumerate(tokens_lower):
            if tk in self.ANAGRAM_INDICATORS:
                found['anagram'].append(i)
            if tk in self.HIDDEN_INDICATORS:
                found['hidden'].append(i)
            if tk in self.CONTAINER_INDICATORS:
                found['container'].append(i)
            if tk in self.REVERSAL_INDICATORS:
                found['reversal'].append(i)
            if tk in self.CHARAD_INDICATORS:
                found['charade'].append(i)
        return dict(found)

    def wordnet_hints(self, tokens: List[str], max_synsets=3):
        hints = {}
        for tok in tokens:
            tok_clean = re.sub(r'[^a-zA-Z\-]', '', tok).lower()
            if not tok_clean:
                hints[tok] = []
                continue
            syns = wn.synsets(tok_clean)
            lemmas = []
            hypernyms = set()
            for s in syns[:max_synsets]:
                lemmas.extend([l.name() for l in s.lemmas()][:3])
                for h in s.hypernyms():
                    hypernyms.add(h.name().split(".")[0])
            hints[tok] = {
                "lemmas": sorted(set(lemmas))[:6],
                "hypernyms": sorted(list(hypernyms))[:6]
            }
        return hints

    def optional_conceptnet(self, token: str, limit=5):
        # Disabled by default (internet). If enabled, returns concept relations from ConceptNet API.
        if not self.use_conceptnet:
            return []
        url = f"https://api.conceptnet.io/c/en/{token}"
        try:
            res = self.requests.get(url, timeout=3).json()
            edges = res.get('edges', [])[:limit]
            relations = []
            for e in edges:
                rel = {
                    "rel": e.get('rel', {}).get('label'),
                    "start": e.get('start', {}).get('label'),
                    "end": e.get('end', {}).get('label'),
                }
                relations.append(rel)
            return relations
        except Exception:
            return []

    def heuristic_fodder_spans(self, doc):
        """
        Heuristic to pick spans likely to be fodder (nouns, noun phrases near indicator words).
        Returns list of (start_idx, end_idx, text)
        """
        spans = []
        for np in doc.noun_chunks:
            spans.append((np.start, np.end-1, np.text))
        # also include adjacent adjectives+noun sequences
        for i, t in enumerate(doc):
            if t.pos_ in ("NOUN", "PROPN"):
                start = i
                end = i
                # include left adjectives
                j = i-1
                while j >= 0 and doc[j].pos_ in ("ADJ","DET","NUM"):
                    start = j
                    j -= 1
                spans.append((start, end, doc[start:end+1].text))
        # dedupe by text
        seen = set()
        out = []
        for s in spans:
            if s[2] not in seen:
                out.append(s)
                seen.add(s[2])
        return out

    def predict_type(self, tokens_lower, indicators_dict, doc):
        """
        Simple priority rules to predict clue type.
        """
        # If explicit anagram indicators present -> anagram
        if indicators_dict.get('anagram'):
            return "anagram"
        if indicators_dict.get('hidden'):
            return "hidden"
        if indicators_dict.get('container'):
            return "container"
        if indicators_dict.get('reversal'):
            return "reversal"
        # if two short noun phrases -> charade
        noun_chunks = list(doc.noun_chunks)
        if len(noun_chunks) >= 2:
            return "charade"
        # fallback to 'definition' or 'unknown'
        # If the clue ends/starts with a dictionary word likely a definition, mark as 'definition'
        if tokens_lower[0] in self.DEFINITION_SEPARATORS or tokens_lower[-1] in self.DEFINITION_SEPARATORS:
            return "definition"
        return "unknown"

    def decompose(self, clue_text: str) -> Dict:
        clue = clue_text.strip()
        length = self.extract_length(clue)  # list of possible lengths or None
        tokens, pos, lemmas, doc = self.pos_and_tokens(clue)
        tokens_lower = [t.lower() for t in tokens]
        indicators = self.detect_indicators(tokens_lower)
        fodder_spans = self.heuristic_fodder_spans(doc)
        predicted_type = self.predict_type(tokens_lower, indicators, doc)
        wn_hints = self.wordnet_hints(tokens)

        # simple selection of likely fodder: nearest noun chunk to first indicator or last noun
        fodder_candidates = []
        if indicators:
            # prefer spans near first indicator
            first_ind_idx = min(min(v) for v in indicators.values())
            # pick spans whose token ranges are near
            for s,e,text in fodder_spans:
                if abs(s - first_ind_idx) <= 4 or abs(e - first_ind_idx) <= 4:
                    fodder_candidates.append((s,e,text))
        if not fodder_candidates and fodder_spans:
            fodder_candidates.append(fodder_spans[0])

        # choose indicator tokens text
        indicator_texts = []
        for typ, idxs in indicators.items():
            for i in idxs:
                if 0 <= i < len(tokens):
                    indicator_texts.append((typ, tokens[i]))

        out = {
            "clue": clue,
            "length": length,
            "tokens": tokens,
            "pos": pos,
            "lemmas": lemmas,
            "predicted_type": predicted_type,
            "indicators": indicator_texts,
            "fodder_candidates": fodder_candidates,  # (start,end,text)
            "wordnet": wn_hints
        }

        # Optionally add conceptnet relations for tokens (disabled by default)
        if self.use_conceptnet:
            cn = {}
            for t in tokens:
                cn[t] = self.optional_conceptnet(t)
            out["conceptnet"] = cn

        return out

    def format_decomposition_prompt(self, decomp: Dict, compact=True) -> str:
        """
        Convert decomposition dict into a short text summary to include in LM prompt.
        compact=True -> one-line summary; else verbose JSON.
        """
        if compact:
            parts = []
            parts.append(f"type={decomp['predicted_type']}")
            if decomp['length']:
                parts.append(f"len={'/'.join(map(str,decomp['length']))}")
            if decomp['indicators']:
                inds = ",".join([f"{t}:{tok}" for t,tok in decomp['indicators']])
                parts.append(f"indicators={inds}")
            if decomp['fodder_candidates']:
                fd = "|".join([f"{span[2]}" for span in decomp['fodder_candidates']])
                parts.append(f"fodder={fd}")
            # small wordnet hint sample
            wn_sample = []
            for k, v in list(decomp['wordnet'].items())[:3]:
                lem = v.get("lemmas", [])[:2]
                hyp = v.get("hypernyms", [])[:1]
                if lem or hyp:
                    wn_sample.append(f"{k}->lem:{','.join(lem)} hyp:{','.join(hyp)}")
            if wn_sample:
                parts.append("wn=" + ";".join(wn_sample))
            return " | ".join(parts)
        else:
            return json.dumps(decomp, ensure_ascii=False)

# ==========================================
# 1. DATA PROCESSING & CURRICULUM SETUP
# ==========================================

class CurriculumDataProcessor:
    @staticmethod
    def scramble_word(word):
        if not isinstance(word, str): return ""
        clean = re.sub(r'[^a-zA-Z]', '', word)
        chars = list(clean)
        random.shuffle(chars)
        return "".join(chars)

    @staticmethod
    def load_isarcasm(path):
        print(f"Loading iSarcasm data from {path}...")
        try:
            df = pd.read_csv(path)
            cols = df.columns.str.lower()
            if 'tweet' in cols: df = df.rename(columns={'tweet': 'input'})
            elif 'text' in cols: df = df.rename(columns={'text': 'input'})

            label_col = 'sarcastic' if 'sarcastic' in cols else 'label'
            processed_data = []
            for _, row in df.iterrows():
                text = str(row['input'])
                label = row[label_col]
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
    def prepare_datasets(train_path, test_path, isarcasm_path):
        print(f"Loading Guardian data from {train_path}...")
        raw_train = pd.read_csv(train_path, dtype=str).dropna()
        raw_test = pd.read_csv(test_path, dtype=str).dropna()

        # Phase 1 Part A - Unscramble
        anagram_data = []
        for tgt in raw_train['target']:
            tgt = str(tgt).strip()
            if len(tgt) > 2:
                scrambled = CurriculumDataProcessor.scramble_word(tgt)
                anagram_data.append({
                    "input": scrambled,
                    "target": tgt,
                    "task_type": "unscramble"
                })
        anagram_df = pd.DataFrame(anagram_data)

        # Phase 1 Part B - Sarcasm
        sarcasm_df = CurriculumDataProcessor.load_isarcasm(isarcasm_path)

        phase1_df = pd.concat([anagram_df, sarcasm_df]).sample(frac=1).reset_index(drop=True)

        train_df = raw_train.copy()
        train_df['task_type'] = "cryptic"

        test_df = raw_test.copy()
        test_df['task_type'] = "cryptic"

        return phase1_df, train_df, test_df

class NeuroCryptDataset(Dataset):
    """
    For T5 (encoder-decoder) we create:
    - input_ids: tokenized input prompt
    - labels: tokenized target (with -100 for padding)
    """

    def __init__(self, dataframe, tokenizer, decomposer: ClueDecomposer, max_len=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.decomposer = decomposer

    def __len__(self):
        return len(self.data)

    def build_prompt(self, inp_text, task):
        # Keep original short prompts but include decomposition for cryptic tasks
        if task == "unscramble":
            return f"unscramble: {inp_text}"
        elif task == "sarcasm":
            return f"classify sarcasm: {inp_text}"
        else:
            match = re.search(r'\((\d+)\)$', inp_text)
            length_val = match.group(1) if match else "unknown"
            return f"solve cryptic: {inp_text} length: {length_val}"

    def __getitem__(self, index):
        row = self.data.iloc[index]
        inp_text = str(row['input']).strip()
        target_text = str(row['target']).strip()
        task = row['task_type']

        # --- Decompose if cryptic ---
        decomp_summary = ""
        if task == "cryptic":
            try:
                decomp = self.decomposer.decompose(inp_text)
                decomp_summary = self.decomposer.format_decomposition_prompt(decomp, compact=True)
            except Exception as e:
                decomp_summary = ""
        
        # Build prompt including decomposition summary
        base_prompt = self.build_prompt(inp_text, task)
        if decomp_summary:
            enhanced_prompt = f"{base_prompt} [DECOMPOSE] {decomp_summary}"
        else:
            enhanced_prompt = base_prompt

        # Tokenize input (encoder)
        input_enc = self.tokenizer(
            enhanced_prompt,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize target (decoder)
        target_enc = self.tokenizer(
            target_text,
            truncation=True,
            max_length=64,  # targets are typically short
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
            'target': target_text
        }

# ==========================================
# 2. T5 + LoRA MODEL WRAPPER
# ==========================================

class NeuroCryptT5:
    def __init__(self, model_name='t5-base', lora_r=8, lora_alpha=16, lora_dropout=0.05, device=None):
        # device selection
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Initializing {model_name} on {self.device} ...")

        # Load tokenizer & model (T5 for conditional generation)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Configure LoRA for T5 (encoder-decoder architecture)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q", "v"],  # T5 uses q, k, v, o in attention
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM"  # Changed for T5
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate_beams(self, raw_input, num_beams=10, max_new_tokens=32):
        match = re.search(r'\((\d+)\)$', raw_input)
        length_val = match.group(1) if match else "unknown"
        input_text = f"solve cryptic: {raw_input} length: {length_val}"

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True
            )
        candidates = [self.tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
        
        return candidates, int(length_val) if length_val != "unknown" else None

# ==========================================
# 3. SYMBOLIC GATEKEEPER & TRAINING
# ==========================================

class SymbolicGatekeeper:
    @staticmethod
    def solve(model_wrapper, raw_input):
        candidates, target_len = model_wrapper.generate_beams(raw_input, num_beams=10)
        valid_answer = None
        for cand in candidates:
            clean_cand = cand.strip().lower()
            # Accept alphabetic answers only
            if not clean_cand.isalpha(): continue
            if target_len is not None:
                if len(clean_cand) != target_len:
                    continue
            valid_answer = clean_cand
            break
        if valid_answer is None:
            valid_answer = candidates[0].strip().lower()
        return valid_answer

def train_epoch(model_wrapper, dataloader, optimizer, desc="Training"):
    model = model_wrapper.model
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=desc)
    for batch in pbar:
        input_ids = batch['input_ids'].to(model_wrapper.device)
        attention_mask = batch['attention_mask'].to(model_wrapper.device)
        labels = batch['labels'].to(model_wrapper.device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():
    BASE_DIR = './LLM_proj/llm_proj_data'
    TRAIN_FILE = os.path.join(BASE_DIR, 'guardian/train.csv')
    TEST_FILE = os.path.join(BASE_DIR, 'guardian/test.csv')
    ISARCASM_FILE = os.path.join(BASE_DIR, 'isarcasm/isarcasm_train.csv')

    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    phase1_df, train_df, test_df = CurriculumDataProcessor.prepare_datasets(TRAIN_FILE, TEST_FILE, ISARCASM_FILE)

    print(f"\n--- DATASET STATISTICS ---")
    print(f"Phase 1 (Sarcasm + Anagrams): {len(phase1_df)} samples")
    print(f"Phase 2 (Cryptic Crosswords): {len(train_df)} samples")
    print(f"Test Set: {len(test_df)} samples")

    # initialize decomposer
    decomposer = ClueDecomposer(spacy_model="en_core_web_sm", use_conceptnet=False)

    # Initialize T5+LoRA model wrapper
    model_name = 't5-base'  # Options: t5-small, t5-base, t5-large, t5-3b
    solver = NeuroCryptT5(model_name=model_name, lora_r=8, lora_alpha=16, lora_dropout=0.05)

    BATCH_SIZE = 8
    MAX_LEN = 256

    phase1_loader = DataLoader(
        NeuroCryptDataset(phase1_df, solver.tokenizer, decomposer, max_len=MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True
    )

    phase2_loader = DataLoader(
        NeuroCryptDataset(train_df, solver.tokenizer, decomposer, max_len=MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # optimizer: only update LoRA/PEFT parameters
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, solver.model.parameters()), lr=3e-4)

    # Phase 1 training
    print("\n=== PHASE 1: FOUNDATION SKILLS (SARCASM & ANAGRAMS) ===")
    for epoch in range(2):
        loss = train_epoch(solver, phase1_loader, optimizer, desc=f"Phase1 Epoch {epoch+1}")
        print(f"Phase 1 Loss: {loss:.4f}")

    # Phase 2 training
    print("\n=== PHASE 2: FINE-TUNING (CRYPTIC CROSSWORDS) ===")
    for epoch in range(2):
        loss = train_epoch(solver, phase2_loader, optimizer, desc=f"Phase2 Epoch {epoch+1}")
        print(f"Phase 2 Loss: {loss:.4f}")

    # Save LoRA adapters (PEFT)
    save_dir = "./t5_lora_adapter"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving LoRA adapter to {save_dir} ...")
    solver.model.save_pretrained(save_dir)

    # Evaluation
    print("\n=== NEURO-SYMBOLIC EVALUATION ON TEST SET ===")
    solver.model.eval()
    correct = 0
    results = []
    eval_data = test_df

    print(f"{'PREDICTION':<15} | {'TARGET':<15} | {'STATUS'}")
    print("-" * 50)
    for _, row in tqdm(eval_data.iterrows(), total=len(eval_data)):
        inp = str(row['input'])
        tgt = str(row['target']).strip().lower()

        pred = SymbolicGatekeeper.solve(solver, inp)
        is_correct = (pred == tgt)
        if is_correct: correct += 1

        results.append({
            "input": inp,
            "target": tgt,
            "prediction": pred,
            "match": is_correct
        })

        if len(results) <= 5:
            print(f"{pred:<15} | {tgt:<15} | {'✅' if is_correct else '❌'}")

    accuracy = (correct / len(eval_data)) * 100
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
    pd.DataFrame(results).to_csv("neurocrypt_t5_final_results.csv", index=False)
    print("Results saved to neurocrypt_t5_final_results.csv")

if __name__ == "__main__":
    main()
