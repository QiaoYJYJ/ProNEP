import os
import argparse
import hashlib

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from configs import get_cfg_defaults
from models import ProNEP
from pro_embedding import embed_sequence
from prose.models.multitask import ProSEMT


MAX_LENGTH_NLR = 1279
MAX_LENGTH_EFF = 1759


def read_fasta(path):
    records = []
    name = None
    seq_parts = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_parts).upper()))

                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)

        if name is not None:
            records.append((name, "".join(seq_parts).upper()))

    return records


def seq_key(seq):
    seq = str(seq).strip().upper()
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


def pad_or_truncate(z, max_length):
    z = z.float()

    if z.size(0) < max_length:
        padding = torch.zeros(
            max_length - z.size(0),
            z.size(1),
            dtype=z.dtype
        )
        z = torch.vstack([z, padding])
    else:
        z = z[:max_length, :]

    return z


def build_embedding_cache(nlr_records, eff_records, cache_path, device):
    if os.path.exists(cache_path):
        print(f"Loading existing embedding cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print("No cache found. Building embedding cache...")

    prose_model = ProSEMT.load_pretrained().to(device)
    prose_model.eval()

    all_sequences = set()

    for _, seq in nlr_records:
        all_sequences.add(seq)

    for _, seq in eff_records:
        all_sequences.add(seq)

    print(f"Unique sequences to embed: {len(all_sequences)}")

    cache = {}

    with torch.inference_mode():
        for i, seq in enumerate(all_sequences):
            key = seq_key(seq)

            z = embed_sequence(
                prose_model,
                seq,
                use_cuda=(device.type == "cuda")
            )

            cache[key] = z.cpu().half()

            if (i + 1) % 50 == 0:
                print(f"Embedded {i + 1}/{len(all_sequences)}")

    torch.save(cache, cache_path)
    print(f"Saved embedding cache to: {cache_path}")

    return cache


class FastaPairDataset(Dataset):
    def __init__(self, nlr_records, eff_records, embedding_cache):
        self.nlr_records = nlr_records
        self.eff_records = eff_records
        self.embedding_cache = embedding_cache

        self.num_nlr = len(nlr_records)
        self.num_eff = len(eff_records)
        self.total_pairs = self.num_nlr * self.num_eff

    def __len__(self):
        return self.total_pairs

    def get_nlr_embedding(self, seq):
        key = seq_key(seq)
        z = self.embedding_cache[key]
        return pad_or_truncate(z, MAX_LENGTH_NLR)

    def get_eff_embedding(self, seq):
        key = seq_key(seq)
        z = self.embedding_cache[key]
        return pad_or_truncate(z, MAX_LENGTH_EFF)

    def __getitem__(self, idx):
        nlr_idx = idx // self.num_eff
        eff_idx = idx % self.num_eff

        nlr_id, nlr_seq = self.nlr_records[nlr_idx]
        eff_id, eff_seq = self.eff_records[eff_idx]

        v_p = self.get_nlr_embedding(nlr_seq)
        v_d = self.get_eff_embedding(eff_seq)

        return nlr_id, eff_id, v_d, v_p


def collate_fn(batch):
    nlr_ids, eff_ids, v_d, v_p = zip(*batch)

    v_d = torch.stack(v_d).float()
    v_p = torch.stack(v_p).float()

    return list(nlr_ids), list(eff_ids), v_d, v_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="config yaml path")
    parser.add_argument("--model", required=True, help="trained ProNEP .pth path")
    parser.add_argument("--nlr_fasta", required=True, help="NLR fasta file")
    parser.add_argument("--eff_fasta", required=True, help="Effector fasta file")
    parser.add_argument("--output", default="fasta_prediction_results.csv")
    parser.add_argument("--cache", default="fasta_embedding_cache.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use, e.g. 0, 1, 2")

    args = parser.parse_args()

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {args.gpu} does not exist. "
                f"Available GPUs: 0-{torch.cuda.device_count() - 1}"
            )

        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)

        print(f"Running on: {device}")
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running on CPU.")

    nlr_records = read_fasta(args.nlr_fasta)
    eff_records = read_fasta(args.eff_fasta)

    print(f"NLR sequences: {len(nlr_records)}")
    print(f"Effector sequences: {len(eff_records)}")
    print(f"Total pairs to predict: {len(nlr_records) * len(eff_records)}")

    embedding_cache = build_embedding_cache(
        nlr_records,
        eff_records,
        args.cache,
        device
    )

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    model = ProNEP(**cfg).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = FastaPairDataset(
        nlr_records,
        eff_records,
        embedding_cache
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    rows = []

    with torch.inference_mode():
        for batch_idx, (nlr_ids, eff_ids, v_d, v_p) in enumerate(loader):
            v_d = v_d.to(device, non_blocking=True)
            v_p = v_p.to(device, non_blocking=True)

            _, _, score, att = model(v_d, v_p, mode="eval")

            probs = torch.sigmoid(score).view(-1).detach().cpu().tolist()
            logits = score.view(-1).detach().cpu().tolist()

            for nlr_id, eff_id, logit, prob in zip(nlr_ids, eff_ids, logits, probs):
                rows.append({
                    "NLR_id": nlr_id,
                    "Effector_id": eff_id,
                    "logit": logit,
                    "interaction_probability": prob,
                    "prediction_0.7384": 1 if prob >= 0.7384 else 0
                })

            print(f"Predicted batch {batch_idx + 1}/{len(loader)}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
