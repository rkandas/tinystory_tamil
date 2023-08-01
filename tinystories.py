"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from time import sleep
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer
from indicTrans.inference.engine import Model
import ftfy.bad_codecs
from datasets import Dataset, DatasetDict

import csv

DATA_CACHE_DIR = "data"
indic2en_model = Model(expdir='./indicTrans/en-indic')

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the dataset to disk."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")

def pretokenize():
    def process_shard(shard):
        total_stories = 0
        print(f"Processing {shard}...")
        with open(shard, "r", encoding='sloppy-windows-1252') as f:
            fulltext = f.read()
            stories = fulltext.split('<|endoftext|>')
            stories = [l.strip() for l in stories]
        current_csv_file = None
        all_translations = []
        for example in tqdm(stories[2300:]):
            total_stories = total_stories + 1
            #text = example["story"]
            text = example.strip()  # get rid of leading/trailing whitespace
            text = text.replace("\n\n", "\n")  # replace newlines with spaces
            text = text.replace("\"", "“")
            #text = text.replace("..", ".")
            if text == "":
                continue

            #tamil_text = ts.translate_text(text, to_language='ta', translator='google')
            tamil_text = indic2en_model.translate_paragraph(text, 'en', 'ta')
            tamil_text = tamil_text.replace("ஃ", ":\n")
            tamil_text = tamil_text.replace("அம்சங்கள்:", "\nஅம்சங்கள்:")
            tamil_text = tamil_text.replace("கதை:", "\nகதை:")
            tamil_text = tamil_text.replace("குறிப்பில்லா வாக்கியம்:", "\nகுறிப்பில்லா வாக்கியம்:")
            tamil_text = tamil_text.replace("சுருக்கம்:", "\nசுருக்கம்:")
            tamil_text = tamil_text.replace("வார்த்தைகள்:", "\nவார்த்தைகள்:")
            tamil_text = tamil_text.replace("சிறப்பம்சங்கள்:", "\nசிறப்பம்சங்கள்:")
            tamil_text = tamil_text.replace("\"\"", "\n")
            #tamil_text = tamil_text.replace(".", ".\n")
            print(tamil_text)
            #sleep(1)
            #except:
                #print("Error in translation.. skipping...")
                #continue

            if current_csv_file is None or len(all_translations) % 100 == 0:
                current_csv_file = shard.replace(".txt", ".csv")
                with open(current_csv_file, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(all_translations)
                all_translations = []

            all_translations.append([tamil_text])

        if current_csv_file is not None:
            with open(current_csv_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_translations)

    # iterate the shards and translate all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    for shard in shard_filenames:
        process_shard(shard)

    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        #shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:

    @staticmethod
    def iter_batches(split, batch_size, max_seq_len, device, num_workers=0):
        ds = PretokDataset(split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=False, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_tokenizer", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    fun = {
        "download": download,
        "pretokenize": pretokenize,
    }
    fun[args.stage]()