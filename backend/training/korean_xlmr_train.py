#!/usr/bin/env python3
"""
Korean XLM-RoBERTa Training Script
Trains Korean hate speech detection model using K-HATERS dataset with XLM-RoBERTa
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# Import our K-HATERS dataset
from khaters_bio_dataset import KhatersBioDataset

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KoreanXLMDetector(nn.Module):
    """Korean hate speech detector with XLM-RoBERTa"""
    
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = AutoModel.from_pretrained(model_name).config
        hidden_size = self.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Task-specific heads
        self.intensity_head = nn.Linear(hidden_size, 4)  # normal, offensive, L1_hate, L2_hate
        self.bio_head = nn.Linear(hidden_size, 5)       # O, B-SOFT, I-SOFT, B-HARD, I-HARD
        self.target_head = nn.Linear(hidden_size, 6)    # gender, age, political, religion, region, others
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Task predictions
        intensity_logits = self.intensity_head(sequence_output[:, 0, :])  # [CLS] token
        bio_logits = self.bio_head(sequence_output)  # [batch_size, seq_len, 5]
        target_logits = self.target_head(sequence_output[:, 0, :])  # [CLS] token
        
        return {
            "intensity_logits": intensity_logits,
            "bio_logits": bio_logits,
            "target_logits": target_logits
        }

class KoreanXLMTrainer:
    """Korean XLM-RoBERTa Model Trainer"""
    
    def __init__(self, model_name: str = "xlm-roberta-base", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"🇰🇷 Initializing Korean XLM-RoBERTa trainer")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Model: {model_name}")
        
        # Load datasets
        self.train_dataset = KhatersBioDataset("train", tokenizer_name=model_name)
        self.test_dataset = KhatersBioDataset("test", tokenizer_name=model_name)
        
        # Create model
        self.model = KoreanXLMDetector(model_name=model_name).to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=16, 
            shuffle=True, 
            collate_fn=self.simple_collate_fn,
            num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=32, 
            shuffle=False, 
            collate_fn=self.simple_collate_fn,
            num_workers=0
        )
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3)
        
        # Loss functions with class weights for Korean data
        self.intensity_loss = nn.CrossEntropyLoss()
        
        # BIO loss with class weights (Korean has different distribution)
        bio_class_weights = torch.tensor([1.0, 8.0, 4.0, 15.0, 8.0]).to(self.device)  # Adjusted for Korean
        self.bio_loss = nn.CrossEntropyLoss(weight=bio_class_weights, ignore_index=-100)
        
        # Target loss for multi-label classification
        self.target_loss = nn.BCEWithLogitsLoss()
        
        print(f"📊 Train samples: {len(self.train_dataset)}")
        print(f"📊 Test samples: {len(self.test_dataset)}")
        print(f"📊 Train batches: {len(self.train_loader)}")
        print(f"📊 Test batches: {len(self.test_loader)}")
        print(f"⚖️ BIO Class weights: {bio_class_weights.cpu().numpy()}")
    
    def simple_collate_fn(self, batch):
        """Simple collate function that only batches tensor data"""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        bio_labels = torch.stack([item["bio_labels"] for item in batch])
        intensity_labels = torch.tensor([item["sentence_labels"].item() for item in batch], dtype=torch.long)
        
        # Handle target labels (multi-label)
        target_labels = torch.stack([item["target_labels"] for item in batch]) if "target_labels" in batch[0] else None
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bio_labels": bio_labels,
            "intensity_labels": intensity_labels
        }
        
        if target_labels is not None:
            result["target_labels"] = target_labels
            
        return result
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_intensity_loss = 0
        total_bio_loss = 0
        total_target_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            bio_labels = batch["bio_labels"].to(self.device)
            intensity_labels = batch["intensity_labels"].to(self.device)
            
            # Handle target labels if available
            target_labels = batch.get("target_labels")
            if target_labels is not None:
                target_labels = target_labels.to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate losses
            intensity_loss = self.intensity_loss(outputs["intensity_logits"], intensity_labels)
            bio_loss = self.bio_loss(outputs["bio_logits"].view(-1, 5), bio_labels.view(-1))
            
            # Target loss (if available)
            if target_labels is not None and "target_logits" in outputs:
                target_loss = self.target_loss(outputs["target_logits"], target_labels)
            else:
                target_loss = torch.tensor(0.0, device=self.device)
            
            # Skip batches with NaN losses
            if torch.isnan(bio_loss) or torch.isnan(intensity_loss):
                print(f"⚠️ Skipping batch {batch_idx} due to NaN loss")
                continue
            
            # Total loss with adjusted weighting for BIO and target
            total_loss_batch = intensity_loss + 2.0 * bio_loss + target_loss  # Include target loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_intensity_loss += intensity_loss.item()
            total_bio_loss += bio_loss.item()
            total_target_loss += target_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'intensity': f'{intensity_loss.item():.4f}',
                'bio': f'{bio_loss.item():.4f}',
                'target': f'{target_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'total_loss': total_loss / num_batches if num_batches > 0 else float('inf'),
            'intensity_loss': total_intensity_loss / num_batches if num_batches > 0 else float('inf'),
            'bio_loss': total_bio_loss / num_batches if num_batches > 0 else float('inf'),
            'target_loss': total_target_loss / num_batches if num_batches > 0 else float('inf')
        }
    
    def evaluate(self):
        """Comprehensive evaluation"""
        self.model.eval()
        
        all_intensity_preds = []
        all_intensity_labels = []
        all_bio_preds = []
        all_bio_labels = []
        all_target_preds = []
        all_target_labels = []
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bio_labels = batch["bio_labels"].to(self.device)
                intensity_labels = batch["intensity_labels"].to(self.device)
                
                # Handle target labels if available
                target_labels = batch.get("target_labels")
                if target_labels is not None:
                    target_labels = target_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate losses
                intensity_loss = self.intensity_loss(outputs["intensity_logits"], intensity_labels)
                bio_loss = self.bio_loss(outputs["bio_logits"].view(-1, 5), bio_labels.view(-1))
                
                # Target loss (if available)
                if target_labels is not None and "target_logits" in outputs:
                    target_loss = self.target_loss(outputs["target_logits"], target_labels)
                else:
                    target_loss = torch.tensor(0.0, device=self.device)
                
                total_loss_batch = intensity_loss + 2.0 * bio_loss + target_loss
                
                # Skip NaN losses
                if torch.isnan(total_loss_batch):
                    continue
                
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                # Get predictions
                intensity_preds = torch.argmax(outputs["intensity_logits"], dim=1)
                bio_preds = torch.argmax(outputs["bio_logits"], dim=2)
                
                # Target predictions (if available)
                if target_labels is not None and "target_logits" in outputs:
                    target_preds = torch.sigmoid(outputs["target_logits"]) > 0.5
                else:
                    target_preds = None
                
                # Collect predictions
                all_intensity_preds.extend(intensity_preds.cpu().numpy())
                all_intensity_labels.extend(intensity_labels.cpu().numpy())
                all_bio_preds.extend(bio_preds.cpu().numpy())
                all_bio_labels.extend(bio_labels.cpu().numpy())
                
                if target_preds is not None and target_labels is not None:
                    all_target_preds.extend(target_preds.cpu().numpy())
                    all_target_labels.extend(target_labels.cpu().numpy())
        
        # Calculate metrics
        intensity_f1 = f1_score(all_intensity_labels, all_intensity_preds, average='weighted')
        intensity_precision = precision_score(all_intensity_labels, all_intensity_preds, average='weighted')
        intensity_recall = recall_score(all_intensity_labels, all_intensity_preds, average='weighted')
        
        # Convert lists to numpy arrays and flatten
        bio_labels_flat = np.array(all_bio_labels).flatten()
        bio_preds_flat = np.array(all_bio_preds).flatten()
        
        # Calculate BIO metrics (excluding padding tokens)
        valid_mask = bio_labels_flat != -100  # Exclude padding
        if np.sum(valid_mask) > 0:
            bio_f1 = f1_score(bio_labels_flat[valid_mask], bio_preds_flat[valid_mask], average='weighted')
            bio_precision = precision_score(bio_labels_flat[valid_mask], bio_preds_flat[valid_mask], average='weighted')
            bio_recall = recall_score(bio_labels_flat[valid_mask], bio_preds_flat[valid_mask], average='weighted')
            
            # Calculate Non-O F1 (harmful spans only)
            non_o_mask = (bio_labels_flat != -100) & (bio_labels_flat != 0)  # Exclude padding and O tags
            if np.sum(non_o_mask) > 0:
                non_o_f1 = f1_score(bio_labels_flat[non_o_mask], bio_preds_flat[non_o_mask], average='weighted')
            else:
                non_o_f1 = 0.0
        else:
            bio_f1 = bio_precision = bio_recall = 0.0
            non_o_f1 = 0.0
        
        # Target metrics (if available)
        if len(all_target_labels) > 0 and len(all_target_preds) > 0:
            target_f1 = f1_score(all_target_labels, all_target_preds, average='macro', zero_division=0)
            target_precision = precision_score(all_target_labels, all_target_preds, average='macro', zero_division=0)
            target_recall = recall_score(all_target_labels, all_target_preds, average='macro', zero_division=0)
        else:
            target_f1 = target_precision = target_recall = 0.0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'loss': avg_loss,
            'intensity': {
                'f1': intensity_f1,
                'precision': intensity_precision,
                'recall': intensity_recall
            },
            'bio': {
                'f1': bio_f1,
                'precision': bio_precision,
                'recall': bio_recall
            },
            'non_o_f1': non_o_f1,
            'target': {
                'f1': target_f1,
                'precision': target_precision,
                'recall': target_recall
            }
        }
    
    def train(self, epochs: int = 5, patience: int = 2):
        """Train the model with Early Stopping"""
        print(f"🏋️ Starting Korean XLM-RoBERTa training for {epochs} epochs with Early Stopping...")
        print(f"⏰ Early Stopping patience: {patience} epochs")
        
        best_f1 = 0
        best_epoch = 0
        patience_counter = 0
        train_history = []
        eval_history = []
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("=" * 60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            train_history.append(train_metrics)
            
            # Evaluate
            eval_metrics = self.evaluate()
            eval_history.append(eval_metrics)
            
            # Print detailed results
            print(f"\n📈 Training Metrics:")
            print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Intensity Loss: {train_metrics['intensity_loss']:.4f}")
            print(f"  BIO Loss: {train_metrics['bio_loss']:.4f}")
            
            print(f"\n📊 Evaluation Metrics:")
            print(f"  Total Loss: {eval_metrics['loss']:.4f}")
            print(f"  Intensity F1: {eval_metrics['intensity']['f1']:.4f}")
            print(f"  BIO F1: {eval_metrics['bio']['f1']:.4f}")
            print(f"  Non-O F1: {eval_metrics['non_o_f1']:.4f}")
            print(f"  Target F1: {eval_metrics['target']['f1']:.4f}")
            
            # Early Stopping logic - Use Non-O F1 for better harmful span detection
            current_f1 = eval_metrics['non_o_f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch + 1
                patience_counter = 0
                self.save_model(f"korean_xlmr_model.pt")
                print(f"💾 Saved best model (Non-O F1: {best_f1:.4f}) at Epoch {best_epoch}")
            else:
                patience_counter += 1
                print(f"⏰ No improvement for {patience_counter} epochs (Best: {best_f1:.4f} at Epoch {best_epoch})")
                
                if patience_counter >= patience:
                    print(f"\n🛑 Early Stopping triggered!")
                    print(f"🏆 Best Non-O F1: {best_f1:.4f} at Epoch {best_epoch}")
                    print(f"📊 Stopped at Epoch {epoch+1} (patience: {patience})")
                    break
        
        # Save training history
        history = {
            'train_history': train_history,
            'eval_history': eval_history,
            'best_non_o_f1': best_f1,
            'best_epoch': best_epoch,
            'total_epochs': len(train_history),
            'early_stopped': patience_counter >= patience,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'model_name': self.model_name,
                'max_epochs': epochs,
                'patience': patience,
                'device': str(self.device),
                'class_weights': [1.0, 8.0, 4.0, 15.0, 8.0]
            }
        }
        
        with open('korean_xlmr_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✅ Korean XLM-RoBERTa training completed!")
        print(f"🏆 Best Non-O F1: {best_f1:.4f} at Epoch {best_epoch}")
        print(f"📊 Total epochs: {len(train_history)}/{epochs}")
        print(f"📁 Model saved: korean_xlmr_model.pt")
        print(f"📁 History saved: korean_xlmr_training_history.json")
        
        return history
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': self.model_name,
                'num_intensity': 4,
                'num_bio_labels': 5
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filename)
        print(f"💾 Model saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Korean XLM-RoBERTa training with Early Stopping")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model name")
    
    args = parser.parse_args()
    
    print("🇰🇷 Starting Korean XLM-RoBERTa Training with Early Stopping...")
    print(f"⚙️ Config: Max {args.epochs} epochs, Patience {args.patience}, {args.model}, {args.device}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = KoreanXLMTrainer(model_name=args.model, device=args.device)
    
    # Train model
    history = trainer.train(epochs=args.epochs, patience=args.patience)
    
    print("🎉 Korean XLM-RoBERTa training completed successfully!")

if __name__ == "__main__":
    main()
