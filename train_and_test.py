import pandas as pd
import numpy as np
import fasttext
import re
import json
import torch
import logging
from accelerate import Accelerator # NEW: Required for half-precision and easy training loop
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import os
import time

# ایجاد دایرکتوری برای ذخیره لاگ‌ها
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# تنظیمات لاگ‌گیری برای ذخیره در فایل و نمایش در کنسول
log_filename = os.path.join(log_dir, f"training_log_{int(time.time())}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())

# 1. تنظیمات (پارامترهای قابل تغییر)
class Config:
    def __init__(self):
        # مسیر فایل‌های داده (حتما فایل‌ها را در همین مسیر قرار دهید)
        self.DATA_PATH_TRAIN = './data/match_data_4030611.csv'
        self.DATA_PATH_TEST = './data/labeled_data.csv'
        
        # تنظیمات مدل BERT
        self.BERT_MODEL_NAME = 'HooshvareLab/bert-fa-base-uncased'
        self.MAX_LEN = 64 #FineTune default=128
        self.BATCH_SIZE = 8 #FineTune default=16
        self.BERT_EPOCHS = 3 #FineTune default=3
        self.LEARNING_RATE = 1e-5
        self.GRADIENT_ACCUMULATION_STEPS = 4 # NEW: Accumulates 4 steps to simulate a batch size of 32 (8*4)
        
        # وزن‌دهی به مدل‌ها در روش Ensemble (BERT با وزن دو برابر)
        self.ENSEMBLE_WEIGHTS = {'bert': 2.0, 'svm': 1.0, 'logreg': 1.0, 'fasttext': 1.0}
        
        # محدود کردن مصرف حافظه GPU به 90%
        self.GPU_USAGE_LIMIT = 0.80 #FineTune default=0.90
        self.USE_HALF_PRECISION = True # NEW: Use torch.float16 or torch.bfloat16 for lower VRAM/heat
        
        # هایپرپارامترهای مدل‌های کلاسیک
        self.SVM_C = 1.0 #FineTune default=1.0 #step-by-step
        self.LOGREG_C = 1.0 
        self.FASTTEXT_EPOCHS = 100 #FineTune default=100
        self.FASTTEXT_LR = 0.1
        self.FASTTEXT_WORDNGRAMS = 6 # Increased to capture more context

config = Config()
logging.info(f"Configuration: {json.dumps(config.__dict__, indent=4)}")

# 2. تنظیم GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
if device.type == 'cuda':
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    max_memory = total_memory * config.GPU_USAGE_LIMIT
    logging.info(f"Total GPU Memory: {total_memory / (1024**3):.2f} GB")
    logging.info(f"Setting CUDA max reserved memory to {max_memory / (1024**3):.2f} GB")
    # این دستور مصرف GPU را به 90% محدود می‌کند
    torch.cuda.set_per_process_memory_fraction(config.GPU_USAGE_LIMIT)

# 3. پیش‌پردازش داده‌ها
def clean_text(text):
    text = str(text)
    # --- تغییر جدید: حذف کاراکترهای خط جدید (\n) ---
    text = text.replace('\n', ' ')
    # حذف URL، هشتگ، نام کاربری و کاراکترهای خاص
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # اطمینان از حذف فاصله اضافی
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

# 4. بارگذاری و آماده‌سازی داده‌ها
logging.info("Loading data...")
try:
    # train_df = pd.read_csv(config.DATA_PATH_TRAIN) #FineTune default
    train_df = pd.read_csv(config.DATA_PATH_TRAIN, nrows=200000)
    test_df = pd.read_csv(config.DATA_PATH_TEST)

    # === Select the 4 annotators you want ===
    annotators = ["user1", "user2", "user3", "user5"]
    # === Count number of votes for class 1 ===
    votes_for_1 = test_df[annotators].sum(axis=1)
    # === Apply your rule ===
    # - final_label = 1 if votes_for_1 is 3 or 4
    # - final_label = 0 if votes_for_1 is 0, 1, or 2
    final_labels = (votes_for_1 >= 3).astype(int)
    # === Add to test_df ===
    test_df["final_label"] = final_labels

    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)
    
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['final_label']  # استفاده از ستون final_label به عنوان لیبل تست
except Exception as e:
    logging.error(f"Error loading or preprocessing data: {e}")
    raise

y_train = y_train.astype(int)
y_test = y_test.astype(int)
logging.info("Data loaded successfully.")

# 5. پیاده‌سازی مدل‌ها

# 5.1. مدل BERT
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_bert():
    logging.info("Training BERT with Gradient Accumulation and Half Precision...")
    # NEW: Initialize Accelerator for managing half-precision and device placement
    accelerator = Accelerator(mixed_precision='fp16' if config.USE_HALF_PRECISION else 'no')

    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(config.BERT_MODEL_NAME, num_labels=2)
    # model.to(device) 

    train_data = BERTDataset(X_train.tolist(), y_train.tolist(), tokenizer, config.MAX_LEN)
    test_data = BERTDataset(X_test.tolist(), y_test.tolist(), tokenizer, config.MAX_LEN)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    # NEW: Prepare all components for Accelerator (handles model.to(device) and half-precision)
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    
    predictions = []
    
    for epoch in range(config.BERT_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.BERT_EPOCHS} Training"):
            input_ids = batch['input_ids']#.to(device)
            attention_mask = batch['attention_mask']#.to(device)
            labels = batch['labels']#.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # --- NEW: GRADIENT ACCUMULATION LOGIC ---
            # Scale loss for accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS 
            # Use accelerator to backpropagate
            accelerator.backward(loss) 
            # Check if it's time to update weights
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS # Un-scale for reporting

            # total_loss += loss.item()
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

        # avg_loss = total_loss / len(train_loader)
        # logging.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model.eval()
    for batch in tqdm(test_loader, desc="BERT Prediction"):
        input_ids = batch['input_ids']#.to(device)
        attention_mask = batch['attention_mask']#.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)

    return np.array(predictions)
"""
# 5.2. مدل‌های کلاسیک (SVM، Logistic Regression) #step-by-step
def train_classic_models():
    logging.info("Training classic models (SVM, Logistic Regression)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # SVM
    svm_model = SVC(kernel='linear', C=config.SVM_C, probability=True)
    svm_model.fit(X_train_vec, y_train)
    svm_preds = svm_model.predict(X_test_vec)
    
    # Logistic Regression
    logreg_model = LogisticRegression(C=config.LOGREG_C, max_iter=1000)
    logreg_model.fit(X_train_vec, y_train)
    logreg_preds = logreg_model.predict(X_test_vec)

    return svm_preds, logreg_preds
    """

# 5.2. مدل‌های کلاسیک (SVM، Logistic Regression) #step-by-step
# این تابع بردارسازی را برای هر دو مدل انجام می‌دهد.
def train_classic_models(X_train, y_train, X_test, y_test, vectorizer=None):
    logging.info("Vectorizing data for classic models...")
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 3)) #FineTune default=5000 without ngram_range #step-by-step
        X_train_vec = vectorizer.fit_transform(X_train)
    else:
        X_train_vec = vectorizer.transform(X_train)
        
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, vectorizer

# تابع برای آموزش و گزارش SVM
def train_svm_model(X_train_vec, X_test_vec, y_train, y_test, config):
    logging.info("Training SVM model...")
    svm_model = SVC(kernel='linear', C=config.SVM_C, class_weight='balanced') #FineTune default kernel='linear' #step-by-step
    svm_model.fit(X_train_vec, y_train)
    svm_preds = svm_model.predict(X_test_vec)
    
    results = evaluate_models(y_test, svm_preds)
    logging.info(f"--- SVM Results ---")
    logging.info(f"Accuracy: {results['accuracy']:.4f}")
    logging.info(f"F1-Score: {results['f1_score']:.4f}")
    return results, svm_preds

# (نیازی به تعریف توابع LogReg، BERT و FastText نیست، اما باید در کد اصلی شما وجود داشته باشند.)














# 5.3. مدل FastText
def train_fasttext():
    logging.info("Training FastText model...")
    train_fasttext_file = 'train_fasttext.txt'
    test_fasttext_file = 'test_fasttext.txt'

    with open(train_fasttext_file, 'w', encoding='utf-8') as f:
        for text, label in zip(X_train, y_train):
            f.write(f"__label__{label} {text}\n")
    
    with open(test_fasttext_file, 'w', encoding='utf-8') as f:
        for text, label in zip(X_test, y_test):
            f.write(f"__label__{label} {text}\n")

    model = fasttext.train_supervised(
        input=train_fasttext_file,
        epoch=config.FASTTEXT_EPOCHS,
        lr=config.FASTTEXT_LR,
        wordNgrams=config.FASTTEXT_WORDNGRAMS,
        verbose=2
    )

    preds = [int(p[0].replace('__label__', '')) for p in model.predict(X_test.tolist())[0]]
    
    os.remove(train_fasttext_file)
    os.remove(test_fasttext_file)

    return np.array(preds)

# 6. مدل Ensemble و ارزیابی
def evaluate_models(y_true, predictions):
    acc = accuracy_score(y_true, predictions)
    prec = precision_score(y_true, predictions, average='macro', zero_division=0)
    rec = recall_score(y_true, predictions, average='macro', zero_division=0)
    f1 = f1_score(y_true, predictions, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}

def ensemble_predictions(preds_dict, weights, y_true):
    logging.info("Combining predictions with Ensemble method...")
    
    total_preds = np.zeros(len(y_true))
    
    for model_name, preds in preds_dict.items():
        weight = weights.get(model_name, 1.0)
        total_preds += preds * weight
    
    final_preds = (total_preds >= (sum(weights.values()) / 2)).astype(int)
    
    return final_preds

# 6.5. ذخیره پیش‌بینی‌های نهایی برای بررسی
def save_predictions_for_review(X_test, y_true, ensemble_preds, filename='ensemble_predictions_review.csv'):
    """
    Combines test text, true labels, and ensemble predictions into a CSV file.
    """
    logging.info(f"Saving final predictions to {filename}...")
    
    # اطمینان از اینکه همه ورودی‌ها به صورت لیست یا آرایه هستند
    data = {
        'Text': X_test.tolist(),
        'True_Label': y_true.tolist(),
        'Ensemble_Prediction': ensemble_preds.tolist()
    }
    
    # ساخت DataFrame و ذخیره به عنوان CSV
    review_df = pd.DataFrame(data)
    
    # ذخیره در همان دایرکتوری لاگ‌ها برای نظم بیشتر
    output_path = os.path.join('logs', filename)
    review_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"Review file successfully saved to: {output_path}")


def get_dynamic_ensemble_weights(results, base_weight=1.0):
    """
    Calculates dynamic ensemble weights based on the F1-Score of each model.

    Args:
        results (dict): Dictionary where keys are model names (e.g., 'bert') 
                        and values are dictionaries containing performance metrics, 
                        including 'f1_score'.
        base_weight (float): The base factor for normalization.

    Returns:
        dict: Dynamic weights for each model.
    """
    
    f1_scores = {name: res['f1_score'] for name, res in results.items()}
    
    if not f1_scores:
        logging.warning("No F1 scores available for dynamic weighting. Using equal weights.")
        return {'bert': 1.0, 'svm': 1.0, 'logreg': 1.0, 'fasttext': 1.0}

    # Find the maximum F1 score to normalize against
    max_f1 = max(f1_scores.values())
    
    # Calculate weights proportional to F1 score, normalized by the max F1
    dynamic_weights = {}
    for name, f1 in f1_scores.items():
        # Weight = (Model F1 / Max F1) * Base Weight
        # If BERT has max F1, its weight will be 1.0 * base_weight
        # If FastText has 0.85 F1 and BERT has 0.88 F1, its weight will be (0.85/0.88) * base_weight
        dynamic_weights[name] = (f1 / max_f1) * base_weight

    logging.info(f"Dynamic Weights Calculated: {dynamic_weights}")
    return dynamic_weights

# 7. جریان اجرای اصلی
if __name__ == "__main__":
    """ #step-by-step
    all_predictions = {}
    
    bert_preds = train_bert()
    all_predictions['bert'] = bert_preds
    
    svm_preds, logreg_preds = train_classic_models()
    all_predictions['svm'] = svm_preds
    all_predictions['logreg'] = logreg_preds
    
    fasttext_preds = train_fasttext()
    all_predictions['fasttext'] = fasttext_preds

    results = {}
    for model_name, preds in all_predictions.items():
        logging.info(f"\n--- {model_name.upper()} Results ---")
        results[model_name] = evaluate_models(y_test, preds)
        logging.info(f"Accuracy: {results[model_name]['accuracy']:.4f}")
        logging.info(f"Precision: {results[model_name]['precision']:.4f}")
        logging.info(f"Recall: {results[model_name]['recall']:.4f}")
        logging.info(f"F1-Score: {results[model_name]['f1_score']:.4f}")
        

    dynamic_weights = get_dynamic_ensemble_weights(results, base_weight=1.0)
    ensemble_preds = ensemble_predictions(all_predictions, dynamic_weights, y_test)
    ensemble_results = evaluate_models(y_test, ensemble_preds)

    logging.info("\n--- ENSEMBLE Results ---")
    logging.info(f"Accuracy: {ensemble_results['accuracy']:.4f}")
    logging.info(f"Precision: {ensemble_results['precision']:.4f}")
    logging.info(f"Recall: {ensemble_results['recall']:.4f}")
    logging.info(f"F1-Score: {ensemble_results['f1_score']:.4f}") """
    results = {} #step-by-step
    all_predictions = {}
    # 1. آماده‌سازی داده‌ها برای مدل‌های کلاسیک (بردارسازی)
    # این تنها مرحله آماده‌سازی مشترک است که باید انجام شود.
    X_train_vec, X_test_vec, vectorizer = train_classic_models(X_train, y_train, X_test, y_test)
    # 2. اجرای SVM (فقط این بخش فعال است)
    svm_results, svm_preds = train_svm_model(X_train_vec, X_test_vec, y_train, y_test, config)
    results['svm'] = svm_results
    all_predictions['svm'] = svm_preds


    # ذخیره گزارش نهایی
    final_report = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'config': config.__dict__,
        'individual_model_results': results,
        # 'ensemble_results': ensemble_results #step-by-step
    }
    
    report_filename = os.path.join(log_dir, f"report_{int(time.time())}.json")
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
        
    logging.info(f"Full report saved to {report_filename}")
    # save_predictions_for_review(X_test, y_test, ensemble_preds, filename=f"review_{int(time.time())}.csv") #step-by-step
    save_predictions_for_review(X_test, y_test, svm_preds, filename=f"review_svm_{int(time.time())}.csv") #step-by-step
