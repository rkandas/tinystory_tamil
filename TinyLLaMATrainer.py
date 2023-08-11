import os
import time
import torch
import numpy as np
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import lightning as L
from lightning.fabric.strategies import SingleDeviceStrategy
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from datasets import load_dataset
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import random
import math

class TinyLLaMATrainer:
    def __init__(self, device, config_file, new_llama_model=False):
        self.model_path = config_file
        self.start_index = 0
        self.device = device

        self.block_size = 512
        self.batch_size = 2
        self.max_iters = 200000
        self.eval_interval = 1000
        self.log_interval = 10
        self.start_index = 0
        self.grad_clip = 1.5
        self.out_dir = "out/training"
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.accumulation_steps = 4

        self.strategy = SingleDeviceStrategy(device="cuda:0", accelerator="cuda")
        self.fabric = L.Fabric(devices=1, precision="bf16-mixed", strategy=self.strategy)
        self.writer = SummaryWriter(log_dir='out/logs')

        self.tokenizer = AutoTokenizer.from_pretrained(config_file, use_fast=True)
        #self.tokenizer.add_tokens(['Q', '%', ')','1','2','3','4','5','6','7','8','9','0','+','-','/','*','P','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','$'])

        self.tokenizer.pad_token = self.tokenizer.eos_token
        if new_llama_model:
            self.lmodel = self.prepare_new_llama_model()
        else:
            self.lmodel = LlamaForCausalLM.from_pretrained(self.model_path)
        self.lmodel.to(device)
        #self.lmodel.resize_token_embeddings(len(self.tokenizer))
        self.fabricmodel = self.fabric.setup_module(self.lmodel)

        if hasattr(self.lmodel, "enable_input_require_grads"):
            self.lmodel.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                 output.requires_grad_(True)
            self.lmodel.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        
        self.optimizer = torch.optim.AdamW(self.fabricmodel.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

        model_size = sum(t.numel() for t in self.fabricmodel.parameters())
        print(f"GPT Model size: {model_size/1000**2:.1f}M parameters")

    def prepare_new_llama_model(self):
        print("Preparing new LLaMA model ...")
        config = LlamaConfig.from_pretrained(self.model_path)
        config.use_cache = False
        #config.torch_dtype = torch.float16
        config.vocab_size = 32000

        print(config)
        model = LlamaForCausalLM(config)
        return model
    

    def load_datasets(self):
        csv_file_path = 'data/TinyStories_all_data/TinyStories-Instruct-train.csv'  # Provide the path to your CSV file
        df = pd.read_csv(csv_file_path)
        self.dataset = df.iloc[:, 0].tolist()   # Get the first column
        random.shuffle(self.dataset)
        self.dataset_iter = iter(self.dataset[0:])

    def load_datasets_merged(self):
        csv_file_path = 'data/tinystories.csv'  # Provide the path to your CSV file
        df = pd.read_csv(csv_file_path)
        self.dataset = df.iloc[:, 0].tolist()   # Get the first column
        random.shuffle(self.dataset)
        self.dataset_iter = iter(self.dataset[0:])
        print(f"Length: {len(self.dataset)}")

    def load_datasets_tamil_eng(self):
        csv_file_path = 'data/thoughtsource_q.csv'  # Provide the path to your CSV file
        df = pd.read_csv(csv_file_path)
        first_column = df.iloc[:, 0].tolist()   # Get the first column
        second_column = df.iloc[:, 1].tolist()  # Get the second column
        self.dataset = first_column + second_column
        random.shuffle(self.dataset)
        self.dataset_iter = iter(self.dataset[0:])
        print(f"Length: {len(self.dataset)}")

    def load_datasets_tamil_eng_q_a(self):
        csv_file_path = 'data/thoughtsource_q.csv'  # Provide the path to your CSV file
        df = pd.read_csv(csv_file_path)
        
        first_column = df.iloc[:, 0].tolist()   # Get the first column
        second_column = df.iloc[:, 1].tolist()  # Get the second column
        
        # Concatenate the first and second columns row-wise
        self.dataset = [f"### Question: {a}\n### Answer:\n{b}" for a, b in zip(first_column, second_column)]
        
        random.shuffle(self.dataset)
        self.dataset_iter = iter(self.dataset[0:])
        print(f"Length: {len(self.dataset)}")

    def text_file_generator(self, file_path):
        context_size = 512
        while True:
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer = ""
                while True:
                    chunk = f.read(context_size)
                    if not chunk:
                        break
                    num_tokens = len(self.tokenizer.tokenize(buffer+chunk))
                    if num_tokens >= self.block_size:
                        yield buffer
                        buffer = chunk
                    else:
                        buffer += chunk

                # Yield the remaining buffer if it's not empty
                if buffer:
                    yield buffer
    
    def hf_generator(self ):
        context_size = self.block_size
        dataset = load_dataset("AnanthZeke/tamil_sentences_master_unique", streaming=True)
        buffer = ""
        for row in dataset['train']:  # Assuming you want to use the 'train' split
            sentence = row['sent_token'] + "\n"
            buffer += sentence
            num_tokens = len(self.tokenizer.tokenize(buffer))
            if num_tokens >= context_size:
                yield buffer[:context_size]
                buffer = buffer[context_size:]

        # Yield the remaining buffer if it's not empty
        if buffer:
            yield buffer
    
    def load_datasets_from_hf(self):
        """
        Load dataset from a text file and split it into chunks based on context_size.
        """
        self.dataset_gen = self.hf_generator()
        print(len(f"Gen: {self.dataset_gen}"))
        self.dataset_iter = iter(self.dataset_gen)

    def load_datasets_from_text(self, file_path):
        """
        Load dataset from a text file and split it into chunks based on context_size.
        """
        self.dataset_gen = self.text_file_generator(file_path)
        print(len(f"Gen: {self.dataset_gen}"))
        self.dataset_iter = iter(self.dataset_gen)
        
    def get_batch(self):
        input_ids_list = []
        targets_list = []

        for i in range(self.batch_size):
            try:
                example = next(self.dataset_iter)
                #example = example.replace("ஃ ",":\n")
                #print(example)
                encoded_example = self.tokenizer(example, truncation=True, max_length=self.block_size, padding="max_length", )
                #print(encoded_example)
                #exit(0)
                #if i>10:
                #    exit()
                input_ids = torch.tensor(encoded_example['input_ids'])
                targets = input_ids.clone()
                targets[:-1] = input_ids[1:]
                targets[-1] = -1
                input_ids_list.append(input_ids)
                targets_list.append(targets)
            except StopIteration:
                self.dataset_iter = iter(self.dataset)
                example = next(self.dataset_iter)
                encoded_example = self.tokenizer(example, truncation=True, max_length=self.block_size, padding="max_length")
                input_ids = torch.tensor(encoded_example['input_ids'])
                targets = input_ids.clone()
                targets[:-1] = input_ids[1:]
                targets[-1] = -1
                input_ids_list.append(input_ids)
                targets_list.append(targets)
            except:
                print(example)
                exit()

        x = torch.stack(input_ids_list, dim=0).to(self.device)
        y = torch.stack(targets_list, dim=0).to(self.device)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf found in input_ids")
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("NaN or Inf found in targets")

        return x, y




    def test_prompt(self, prompt: str) -> None:
         input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
         attention_mask = torch.ones_like(input_ids).to(self.device)
         with torch.no_grad():
            output = self.lmodel.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256, temperature=0.9, top_p=0.98, top_k=50, repetition_penalty = 2.0, do_sample=True,output_scores=False, pad_token_id=self.tokenizer.eos_token_id)
            output_str = self.tokenizer.decode(output[0], skip_special_tokens=False)
            print(output_str)

    def calculate_loss(self, input_ids, targets, model, fabric):

        logits = model(input_ids, return_dict=True).logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        fabric.backward(loss)
        return loss, logits

    def calculate_accuracy_and_average_loss_log_progress(self, iter_num, log_interval, total_loss, logits, targets, t0):
        avg_loss = total_loss / log_interval if iter_num != 0 else total_loss
        dt = time.time() - t0
        predicted_labels = torch.argmax(logits, dim=-1)
        correct_predictions = torch.sum(predicted_labels == targets).detach()
        total_predictions = targets.numel()
        accuracy = correct_predictions / total_predictions
        print(f"iter {iter_num}: loss {avg_loss:.4f}, accuracy: {accuracy:.4f}, time: {dt*1000:.2f}ms")
        self.writer.add_scalar('Loss/train',avg_loss, iter_num)
        self.writer.add_scalar('Accuracy/train', accuracy, iter_num)
        self.writer.flush()
        return accuracy

    def train(self, fabric, train_data):
        iter_num = 0
        total_loss = 0
        t0 = time.time()

        while iter_num <= self.max_iters:
            if iter_num % self.eval_interval == 0:
                if iter_num > 0:
                    self.save_checkpoint()

            input_ids, targets = self.get_batch()

            loss, logits = self.calculate_loss(input_ids, targets, self.fabricmodel, fabric)
            
            if math.isnan(loss):
                print(f"Loss is NaN. Skipping this iteration.")
                continue

            total_loss += loss.detach()

            if math.isnan(loss) or math.isnan(total_loss): # Model went Burrrrrr. Terminate the training.
                exit()
                
            torch.nn.utils.clip_grad_norm_(self.fabricmodel.parameters(), max_norm=1.0)
        
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if iter_num % self.log_interval == 0:
                self.calculate_accuracy_and_average_loss_log_progress(iter_num, self.log_interval, total_loss, logits, targets, t0)
                total_loss = 0
                t0 = time.time()
                
            iter_num += 1

    def save_checkpoint(self):
        print(f"Saving checkpoint to {self.model_path}")
        self.lmodel.save_pretrained(self.model_path)
        print("Model saved to ", self.model_path)


def init_device_and_fabric():
    device = "cuda" if torch.has_cuda else exit("No cuda detected. Aborting the training.")
    strategy = SingleDeviceStrategy(device="cuda:0", accelerator="cuda")
    fabric = L.Fabric(devices=1, precision="bf16-mixed", strategy=strategy)
    writer = SummaryWriter(log_dir='out/logs')
    fabric.seed_everything(1337 + fabric.global_rank)
    fabric.launch()

    return device, fabric, writer

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    device, fabric, writer = init_device_and_fabric()

    trainer = TinyLLaMATrainer(device, "./config_25m", new_llama_model=True)
    #train_data = trainer.load_datasets_merged()
    #train_data = trainer.load_datasets_from_text("data/ponniyin.txt")
    train_data = trainer.load_datasets_from_hf()

    trainer.train(fabric, train_data)
    writer.close()