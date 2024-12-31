import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import math
from transformers import GPT2Tokenizer

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand_as(attn_scores)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
       
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
       
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_positional_encoding(self, max_seq_len, d_model):
        pos_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        if tgt is None:
            tgt = src
            
       
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)
            
        
        src = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)
        
        
        src = src + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt = tgt + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        
        
        src = self.dropout(src)
        tgt = self.dropout(tgt)
        
       
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        
        dec_output = tgt
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask, src_mask)
            
        
        output = self.final_layer(dec_output)
        return output

class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading data from {text_path}")
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        
        self.qa_pairs = []
        pairs = content.split('\n\n')
        for pair in pairs:
            try:
                question = pair.split('Question: ')[1].split('\nAnswer: ')[0]
                answer = pair.split('Answer: ')[1]
                self.qa_pairs.append((question, answer))
            except:
                continue
        
        print(f"Loaded {len(self.qa_pairs)} QA pairs")

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        
        # Format as: "Question: {question} Answer: {answer}"
        input_text = f"Question: {question} Answer: "
        target_text = answer
        
        # Tokenize input
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'target_ids': target_encodings['input_ids'].squeeze(),
            'target_attention_mask': target_encodings['attention_mask'].squeeze()
        }

def train_model(transformer, dataloader, optimizer, scheduler, device, epochs, tokenizer):
    transformer.train()
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    best_loss = float('inf')
    
    print(f"Training on {device}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            src_mask = batch['attention_mask'].to(device)
            tgt_mask = batch['target_attention_mask'].to(device)
            
            # Forward pass
            outputs = transformer(
                src=input_ids,
                tgt=target_ids,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            mask = target_ids != tokenizer.pad_token_id
            correct_predictions += (predictions[mask] == target_ids[mask]).sum().item()
            total_predictions += mask.sum().item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(progress_bar.n+1):.4f}',
                'accuracy': f'{accuracy:.4f}'
            })
        
        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"\nEpoch {epoch+1} Statistics:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(transformer.state_dict(), "transformer_best.pth")
        
        # Regular checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(transformer.state_dict(), f"transformer_checkpoint_epoch_{epoch+1}.pth")

def main():
    # Model parameters
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    input_vocab_size = 50257  # GPT-2 vocabulary size
    target_vocab_size = 50257
    max_seq_len = 128
    dropout = 0.1
    
    # Training parameters
    learning_rate = 3e-4
    batch_size = 16
    epochs = 20
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloader
    dataset = TextDataset("training_data.txt", tokenizer, max_seq_len)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0
    )
    
    # Initialize model
    transformer = Transformer(
        num_layers, d_model, num_heads, d_ff,
        input_vocab_size, target_vocab_size, max_seq_len, dropout
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(transformer.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader))
    
    # Train model
    try:
        train_model(transformer, dataloader, optimizer, scheduler, device, epochs, tokenizer)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining interrupted: {str(e)}")
    finally:
        # Save final model
        torch.save(transformer.state_dict(), "transformer_final.pth")
        print("Model saved!")

if __name__ == "__main__":
    main()
