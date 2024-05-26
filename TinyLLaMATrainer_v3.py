import os
import time
import torch
import numpy as np
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import lightning as L
from lightning.fabric.strategies import SingleDeviceStrategy
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List, Iterator
from datasets import load_dataset
import pandas as pd
import random

class DatasetLoader:
    def __init__(self, tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_source = None
        self.load_method = None
        self.load_args = None
        self.data_iterator = None

    def _load_csv(self, file_path: str, columns: List[int], combine_rows_to_blocksize=False) -> List[str]:
        df = pd.read_csv(file_path, encoding="utf-8")
        data = []
        # print total number of rows in the dataframe
        print("Total number of rows: {}".format(len(df)))
        
        for col in columns:
            data.extend(df.iloc[:, col].tolist())
        buffer = ""
        if not combine_rows_to_blocksize:
            random.shuffle(data)
            for row in data:
                buffer = (row + "\n\n")
                yield buffer
        else:
            for row in data:
                sentence = row  + "\n\n"
                num_tokens = len(self.tokenizer.tokenize(buffer))
                if num_tokens >= self.block_size:
                    yield buffer
                    buffer = sentence
                else:
                    buffer += sentence

        # Yield the remaining buffer if it's not empty
        if buffer:
            yield buffer

    def load_from_csv(self, file_path: str, columns: List[int] = [0], combine_rows_to_blocksize=False) -> Iterator[str]:
        self.data_source = self._load_csv(file_path, columns,combine_rows_to_blocksize)
        self.load_method = self.load_from_csv
        self.load_args = (file_path, columns)
        self.data_iterator = iter(self.data_source)
        return self.data_iterator

    def load_from_text(self, file_path: str) -> Iterator[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
        self.load_method = self.load_from_text
        self.load_args = (file_path,)
        self.data_iterator = iter(self.data_source)
        return self.data_iterator
    
    def hf_generator(self, dataset_name: str, split: str = 'train', streaming: bool = True, text_column_name: str = "Text"):
        context_size = self.block_size
        self.dataset = load_dataset(dataset_name, streaming=streaming)
        buffer = ""
        for row in self.dataset[split]:
            sentence = row[text_column_name] + "</s>\n<s>"
            num_tokens = len(self.tokenizer.tokenize(buffer))
            if num_tokens >= context_size:
                yield buffer
                buffer = sentence
            else:
                buffer += sentence

        # Yield the remaining buffer if it's not empty
        if buffer:
            yield buffer

    def load_from_hf(self, dataset_name: str, split: str = 'train', streaming: bool = True, text_column_name: str = "Text") -> Iterator[str]:
        self.data_source = self.hf_generator(dataset_name, split, streaming, text_column_name)
        self.load_method = self.load_from_hf
        self.load_args = (dataset_name, split, streaming, text_column_name)
        self.data_iterator = iter(self.data_source)
        return self.data_iterator


    def reset(self):
        if self.load_method:
            self.load_method(*self.load_args)


class TinyLLaMATrainer:
    def __init__(self, device, config_file, new_llama_model=False):
        self.model_path = config_file
        self.device = device
        self.setup_hyperparameters()
        self.strategy = SingleDeviceStrategy(device=f"{device}:0", accelerator=device)
        self.fabric = L.Fabric(devices=1, precision="bf16-mixed", strategy=self.strategy)
        self.writer = SummaryWriter(log_dir='out/logs')
        self.tokenizer = AutoTokenizer.from_pretrained(config_file, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lmodel = self.prepare_llama_model(new_llama_model)
        self.lmodel.to(device)
        self.fabricmodel = self.fabric.setup_module(self.lmodel)
        self.setup_hooks()
        self.optimizer = self.setup_optimizer()
        model_size = sum(t.numel() for t in self.fabricmodel.parameters())
        print(f"GPT Model size: {model_size/1000**2:.1f}M parameters")

    def setup_hyperparameters(self):
        self.block_size = 1024
        self.batch_size = 4
        self.max_iters = 320000
        self.eval_interval = 1000
        self.log_interval = 10
        self.grad_clip = 1.5
        self.out_dir = "out/tamilbase"
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.accumulation_steps = 8
        self.lr_scheduler = {
        "type": "cosine",
        "warmup_steps": 1000,
        "total_steps": self.max_iters
        }

    def prepare_llama_model(self, new_llama_model: bool) -> LlamaForCausalLM:
        if new_llama_model:
            config = LlamaConfig.from_pretrained(self.model_path)
            config.use_cache = False
            #config.vocab_size = 32000
            config.torch_dtype = torch.float16
            print(config)
            
            return LlamaForCausalLM(config)
        return LlamaForCausalLM.from_pretrained(self.model_path)

    def setup_hooks(self):
        if hasattr(self.lmodel, "enable_input_require_grads"):
            self.lmodel.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.lmodel.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def setup_optimizer(self):
        optimizer = torch.optim.AdamW(self.fabricmodel.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return self.fabric.setup_optimizers(optimizer)
    
    def encode_data(self, data: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_data = self.tokenizer(data, truncation=True, max_length=self.block_size, padding="max_length", add_special_tokens=True)
        input_ids = torch.tensor(encoded_data['input_ids'])
        targets = input_ids.clone()
        targets[:-1] = input_ids[1:]
        targets[-1] = -1
        return input_ids, targets

    def get_batch(self, dataset_loader: DatasetLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids_list = []
        targets_list = []

        for _ in range(self.batch_size):
            try:
                example = next(dataset_loader)
                #print(example)
                input_ids, targets = self.encode_data(example)
                #print(input_ids)
                input_ids_list.append(input_ids)
                targets_list.append(targets)
            except StopIteration:
                # Handle the end of the dataset
                pass

        if self.device == "mps":
            x = torch.stack(input_ids_list, dim=0).to(self.device, non_blocking=True)
            y = torch.stack(targets_list, dim=0).to(self.device, non_blocking=True)
        elif self.device == "cuda":
            x = torch.stack(input_ids_list, dim=0).pin_memory().to(self.device, non_blocking=True)
            y = torch.stack(targets_list, dim=0).pin_memory().to(self.device, non_blocking=True)
        else:
            x = torch.stack(input_ids_list, dim=0).to(self.device)
            y = torch.stack(targets_list, dim=0).to(self.device)
        return x, y
    
    def calculate_loss(self, input_ids, targets, model, fabric):
        logits = model(input_ids, return_dict=True).logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        fabric.backward(loss)
        return loss, logits

    def train(self, dataset_loader: DatasetLoader):
        iter_num = 0
        total_loss = 0
        t0 = time.time()

        while iter_num <= self.max_iters:
            if iter_num % self.eval_interval == 0 and iter_num > 0:
                self.save_checkpoint()

            input_ids, targets = self.get_batch(dataset_loader)
            loss, logits = self.calculate_loss(input_ids, targets=targets, model=self.fabricmodel, fabric=self.fabric)
            #loss = loss.item() / self.accumulation_steps
            #loss.backward()

            total_loss += loss.item()

            if iter_num % self.log_interval == 0 and iter_num > 0:
                elapsed = time.time() - t0
                self.writer.add_scalar('Loss/train', total_loss / self.log_interval, iter_num)
                print(f"Iteration {iter_num} | Loss {total_loss / self.log_interval:.2f} | Elapsed {elapsed:.2f} sec")
                total_loss = 0
                t0 = time.time()

            if iter_num % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.fabricmodel.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            iter_num += 1

        self.save_checkpoint()

    def save_checkpoint(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.fabricmodel.save_pretrained(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "mps"
    trainer = TinyLLaMATrainer(device, "tamilbase",new_llama_model=True)
    dataset_loader = DatasetLoader(trainer.tokenizer, trainer.block_size)
    #data_iterator = dataset_loader.load_from_hf("AnanthZeke/tamil_sentences_master_unique",text_column_name="sent_token")
    data_iterator = dataset_loader.load_from_csv("./data/processed_content.csv",combine_rows_to_blocksize=True)
    #data_iterator = dataset_loader.load_from_text("./data/tamil_sentences.txt")
    trainer.train(data_iterator)
