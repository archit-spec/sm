#!/usr/bin/env python3
"""
evals.py - batched recall evaluation using vLLM with robust response extraction and debug logging.

Saves:
 - recall_out/masked_completion_results.csv/.json (metrics)
 - recall_out/masked_completion_outputs.json (detailed generations)
 - recall_out/file_completion_results.csv/.json (metrics)
 - recall_out/file_completion_outputs.json (detailed generations)
 - recall_out/aggregate.json

Debug raw engine responses (one JSON line per response) are appended to:
/tmp/vllm_debug_responses.jsonl

Usage example:
python3 evals.py \
  --repo_dir archit11/hyperswitch-code-only \
  --model_path archit11/qwen-4b-hyperswitch-v1 \
  --out_dir ./recall_out \
  --k 3 \
  --ext_filter .rs,.res \
  --max_files 200 \
  --batch_size 8 \
  --top_p 0.95 \
  --temperature 0.2 \
  --max_ref_len 2048 \
  --max_gen_tokens 2048 \
  --shuffle
"""
import argparse
import json
import random
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from tqdm import tqdm

# optional HF dataset loader
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

# metrics
from rouge_score import rouge_scorer
import sacrebleu

# tokenizer
from transformers import AutoTokenizer

# vLLM
try:
    from vllm import LLM, SamplingParams
except Exception as e:
    raise ImportError("vllm not installed or available. Install with `pip install vllm`.") from e


# ---------------- utils ----------------
def is_hf_dataset_id(path: str) -> bool:
    return "/" in path and load_dataset is not None


def load_files_from_local_dir(repo_dir: str) -> List[Dict[str, Any]]:
    p = Path(repo_dir)
    files = []
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        suffix = f.suffix
        if suffix == "" and f.name in ("Dockerfile", "Makefile", "justfile"):
            suffix = f.name
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""
        files.append({"file": str(f), "content": content, "ext": suffix})
    return files


def load_files_from_hf_dataset(dataset_id: str, split: str = "train") -> List[Dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("datasets library not installed.")
    ds = load_dataset(dataset_id, split=split)
    files = []
    for row in ds:
        content = row.get("content") or row.get("text") or ""
        file_path = row.get("file_path") or row.get("path") or row.get("name") or "<row>"
        ext = row.get("extension") or Path(file_path).suffix
        files.append({"file": str(file_path), "content": content, "ext": ext})
    return files


def simple_rust_mask(text: str, max_masks: int = 5) -> List[Tuple[str, str, Tuple[int, int]]]:
    out = []
    i = 0
    n = len(text)
    while len(out) < max_masks:
        idx = text.find("fn ", i)
        if idx == -1:
            break
        brace_idx = text.find("{", idx)
        if brace_idx == -1:
            i = idx + 3
            continue
        depth = 0
        j = brace_idx
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= n:
            i = idx + 3
            continue
        prompt = text[: brace_idx + 1]
        target = text[brace_idx + 1 : j]
        out.append((prompt, target, (brace_idx + 1, j)))
        i = j + 1
    return out


def file_completion_mask(text: str, prefix_frac: float = 0.2) -> Tuple[str, str]:
    L = len(text)
    split = max(32, int(L * prefix_frac))
    prompt = text[:split]
    target = text[split:]
    return prompt, target


# ---------------- metrics ----------------
def token_recall(tokenizer, reference: str, candidate: str) -> float:
    try:
        ref_ids = tokenizer.encode(reference)
        cand_ids = set(tokenizer.encode(candidate))
    except Exception:
        ref_tokens = reference.split()
        cand_tokens = set(candidate.split())
        if not ref_tokens:
            return 0.0
        return sum(1 for t in ref_tokens if t in cand_tokens) / len(ref_tokens)
    if len(ref_ids) == 0:
        return 0.0
    recovered = sum(1 for t in ref_ids if t in cand_ids)
    return recovered / len(ref_ids)


def exact_match(reference: str, candidate: str) -> int:
    return int(reference.strip() == candidate.strip())


def bleu_score(reference: str, candidate: str) -> float:
    try:
        bleu = sacrebleu.sentence_bleu(candidate, [reference])
        return float(bleu.score)
    except Exception:
        return 0.0


def rouge_l(reference: str, candidate: str) -> float:
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        sc = scorer.score(reference, candidate)
        return float(sc["rougeL"].fmeasure * 100.0)
    except Exception:
        try:
            a = reference or ""
            b = candidate or ""
            if not a or not b:
                return 0.0
            la, lb = len(a), len(b)
            best = 0
            max_len_check = min(1024, la, lb)
            for i in range(0, la):
                if la - i <= best:
                    break
                for j in range(i + best + 1, min(la, i + max_len_check) + 1):
                    sub = a[i:j]
                    if sub in b:
                        best = max(best, j - i)
            return float(best) / max(1, min(len(reference), len(candidate))) * 100.0
        except Exception:
            return 0.0


# ---------------- vLLM wrapper (robust + debug) ----------------
class VLLMWrapper:
    def __init__(self, model_path: str, device: str = "cuda"):
        # create engine
        self.model = LLM(model=model_path)

        # try to detect model max sequence length via available attributes; fallback to 40960
        self.model_max_seq_len = None
        try:
            conf = getattr(self.model, "model", None)
            if conf is not None and hasattr(conf, "max_seq_len"):
                self.model_max_seq_len = int(getattr(conf, "max_seq_len"))
        except Exception:
            self.model_max_seq_len = None
        if self.model_max_seq_len is None:
            try:
                self.model_max_seq_len = int(getattr(self.model, "max_seq_len", None) or getattr(self.model, "max_seq_len", 0) or None)
            except Exception:
                self.model_max_seq_len = None
        if not self.model_max_seq_len:
            self.model_max_seq_len = 40960

    # safe-serialize helper for debugging
    def _safe_serialize(self, obj):
        try:
            json.dumps(obj)
            return obj
        except Exception:
            pass
        out = {}
        try:
            out["repr"] = repr(obj)[:4000]
        except Exception:
            out["repr"] = str(type(obj))
        for attr in ("id", "request_id", "request", "text", "outputs", "generated_text", "choices"):
            try:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    try:
                        json.dumps(val)
                        out[attr] = val
                    except Exception:
                        out[attr] = repr(val)[:1000]
            except Exception:
                pass
        return out

    def _dump_debug(self, obj, note=None):
        try:
            rec = {"ts": time.time(), "note": note, "raw": self._safe_serialize(obj)}
            with open("/tmp/vllm_debug_responses.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _extract_text_from_response(self, resp) -> str:
        if resp is None:
            return ""
        # try many places where text may live
        try:
            if hasattr(resp, "text"):
                return getattr(resp, "text") or ""
        except Exception:
            pass
        try:
            if hasattr(resp, "generated_text"):
                return getattr(resp, "generated_text") or ""
        except Exception:
            pass
        try:
            if hasattr(resp, "outputs"):
                outs = getattr(resp, "outputs")
                if isinstance(outs, (list, tuple)) and len(outs) > 0:
                    first = outs[0]
                    if hasattr(first, "text"):
                        return getattr(first, "text") or ""
                    if hasattr(first, "generated_text"):
                        return getattr(first, "generated_text") or ""
        except Exception:
            pass
        try:
            # some libs put choices
            if hasattr(resp, "choices"):
                choices = getattr(resp, "choices")
                if isinstance(choices, (list, tuple)) and choices:
                    c0 = choices[0]
                    if isinstance(c0, dict) and "text" in c0:
                        return c0.get("text") or ""
                    if hasattr(c0, "text"):
                        return getattr(c0, "text") or ""
        except Exception:
            pass
        # fallback to string conversion
        try:
            return str(resp)
        except Exception:
            return ""

    def generate_batch(self, prompts: List[str], max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.95) -> List[str]:
        """
        Robust batched generation wrapper with extensive response extraction and debug logging.
        Writes raw responses to /tmp/vllm_debug_responses.jsonl for inspection.
        """
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        gen_iter = None
        tried = []
        # try batch list signature
        try:
            tried.append("prompts_list")
            gen_iter = self.model.generate(prompts, sampling_params=sampling_params)
        except Exception as e1:
            self._dump_debug({"error": repr(e1), "stage": "prompts_list"}, note="generate failure")
            # try requests kw
            try:
                tried.append("requests_kw")
                requests = [{"id": str(i), "prompt": p} for i, p in enumerate(prompts)]
                gen_iter = self.model.generate(requests=requests, sampling_params=sampling_params)
            except Exception as e2:
                self._dump_debug({"error": repr(e2), "stage": "requests_kw"}, note="generate failure")
                # try positional single list
                try:
                    tried.append("prompts_positional")
                    gen_iter = self.model.generate(prompts)
                except Exception as e3:
                    self._dump_debug({"error": repr(e3), "stage": "prompts_positional"}, note="generate failure")
                    # fallback to per-prompt generation
                    tried.append("per_prompt_fallback")
                    results = []
                    for p in prompts:
                        try:
                            single_txt = ""
                            try:
                                it = self.model.generate([p], sampling_params=sampling_params)
                            except Exception:
                                it = self.model.generate(p)
                            for r in it:
                                # try to extract text from r and from elements
                                if isinstance(r, (list, tuple)):
                                    cand = r[0]
                                else:
                                    cand = r
                                txt = self._extract_text_from_response(cand)
                                if txt:
                                    single_txt = txt
                                    break
                            results.append(single_txt or "")
                            self._dump_debug({"prompt": p, "result": single_txt}, note="per_prompt_item")
                        except Exception as e_p:
                            results.append("")
                            self._dump_debug({"error": repr(e_p), "prompt": p}, note="per_prompt_exception")
                    # record what we tried
                    self._dump_debug({"tried": tried}, note="per_prompt_return")
                    return results

        # if we have an iterator, collect outputs
        results = [""] * len(prompts)
        collected = 0
        try:
            for resp in gen_iter:
                # debug dump raw resp
                self._dump_debug(resp, note="iter_resp")
                # normalize candidate
                candidate = None
                if isinstance(resp, (list, tuple)):
                    # choose first non-none element
                    candidate = None
                    for r in resp:
                        if r is None:
                            continue
                        candidate = r
                        break
                else:
                    candidate = resp
                txt = self._extract_text_from_response(candidate)
                # try to find id mapping
                rid = None
                try:
                    if hasattr(candidate, "request_id"):
                        rid = getattr(candidate, "request_id")
                    elif hasattr(candidate, "id"):
                        rid = getattr(candidate, "id")
                except Exception:
                    rid = None
                if rid is not None:
                    try:
                        idx = int(rid)
                        if 0 <= idx < len(prompts) and not results[idx] and txt:
                            results[idx] = txt
                            collected += 1
                    except Exception:
                        pass
                else:
                    # fill first empty slot
                    if txt:
                        for i in range(len(results)):
                            if results[i] == "":
                                results[i] = txt
                                collected += 1
                                break
                if collected >= len(prompts):
                    break
        except Exception as e_iter:
            # log iteration error
            self._dump_debug({"error": repr(e_iter)}, note="iter_exception")
            try:
                # final fallback: try to extract from gen_iter itself
                txt = self._extract_text_from_response(gen_iter)
                if txt:
                    return [txt] * len(prompts)
            except Exception:
                pass
        # final debug note
        self._dump_debug({"tried": tried, "collected": collected, "results_preview": [r[:200] for r in results]}, note="final_results")
        return results


# ---------------- evaluation (batched) ----------------
def tail_trim_prompt(prompt_text: str, tokenizer, max_context_tokens: int) -> str:
    """Keep the last max_context_tokens tokens of prompt_text (tokenizer-based)."""
    try:
        toks = tokenizer.encode(prompt_text)
        if len(toks) <= max_context_tokens:
            return prompt_text
        trimmed = toks[-max_context_tokens:]
        try:
            return tokenizer.decode(trimmed)
        except Exception:
            return " ".join(map(str, trimmed))
    except Exception:
        return prompt_text[-max_context_tokens * 4 :]


def eval_masked_completion_batched(files, vllm_client: VLLMWrapper, tokenizer, out_dir: Path,
                                  k: int = 3, batch_size: int = 8, top_p: float = 0.95, temperature: float = 0.2,
                                  max_gen_tokens: int = 1024):
    records = []
    mask_items = []
    for f in files:
        txt = f.get("content") or ""
        if not txt.strip():
            continue
        masks = simple_rust_mask(txt, max_masks=3)
        for idx, (prompt, target, (_s, _e)) in enumerate(masks):
            mask_items.append((prompt, target, f["file"], idx))

    total = len(mask_items)
    if total == 0:
        return pd.DataFrame(records)

    model_max = vllm_client.model_max_seq_len
    max_context_tokens = max(256, model_max - max_gen_tokens)

    pbar = tqdm(range(0, total, batch_size), desc="masked_completion_batches")
    for start in pbar:
        end = min(start + batch_size, total)
        batch = mask_items[start:end]
        prompts = []
        for (prompt, target, file_path, mask_idx) in batch:
            trimmed_prompt = tail_trim_prompt(prompt, tokenizer, max_context_tokens)
            prompts.append(trimmed_prompt)
        candidates_per_prompt = [[] for _ in prompts]
        for _ in range(k):
            outs = vllm_client.generate_batch(prompts, max_tokens=min(1024, max_gen_tokens), temperature=temperature, top_p=top_p)
            for i, out in enumerate(outs):
                candidates_per_prompt[i].append(out)
        for i, (prompt, target, file_path, mask_idx) in enumerate(batch):
            cands = candidates_per_prompt[i]
            best_bleu = best_rec = best_em = best_rouge = 0.0
            for c in cands:
                rec = token_recall(tokenizer, target, c)
                bl = bleu_score(target, c)
                rl = rouge_l(target, c)
                em = exact_match(target, c)
                if bl > best_bleu:
                    best_bleu = bl
                if rec > best_rec:
                    best_rec = rec
                if em > best_em:
                    best_em = em
                if rl > best_rouge:
                    best_rouge = rl
            records.append({
                "file": file_path,
                "mask_idx": mask_idx,
                "target_len": len(target),
                "prompt_snippet": (prompt[:400] + "...") if len(prompt) > 400 else prompt,
                "target_snippet": (target[:400] + "...") if len(target) > 400 else target,
                "generations": cands,
                "best_token_recall": best_rec,
                "best_bleu": best_bleu,
                "best_rougeL": best_rouge,
                "exact_match_any": int(best_em),
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(out_dir / "masked_completion_results.csv", index=False)
        (out_dir / "masked_completion_results.json").write_text(df.to_json(orient="records"))
    else:
        (out_dir / "masked_completion_results.csv").write_text("")
        (out_dir / "masked_completion_results.json").write_text("[]")
    (out_dir / "masked_completion_outputs.json").write_text(json.dumps(records, indent=2))
    return df


def eval_file_completion_batched(files, vllm_client: VLLMWrapper, tokenizer, out_dir: Path,
                                k: int = 3, batch_size: int = 8, top_p: float = 0.95, temperature: float = 0.2,
                                max_gen_tokens: int = 2048, max_ref_len: int = 2048):
    records = []
    items = []
    for f in files:
        txt = f.get("content") or ""
        if len(txt.strip()) < 64:
            continue
        prompt, target = file_completion_mask(txt, prefix_frac=0.2)

        # token-aware truncation of reference (so scoring matches tokenizer)
        try:
            ref_ids = tokenizer.encode(target)
            if isinstance(ref_ids, (list, tuple)) and len(ref_ids) > max_ref_len:
                try:
                    target = tokenizer.decode(ref_ids[:max_ref_len])
                except Exception:
                    target = " ".join(map(str, ref_ids[:max_ref_len]))
        except Exception:
            target = target[:max_ref_len]

        items.append((prompt, target, f["file"]))

    total = len(items)
    if total == 0:
        return pd.DataFrame(records)

    model_max = vllm_client.model_max_seq_len
    max_context_tokens = max(256, model_max - max_gen_tokens)

    pbar = tqdm(range(0, total, batch_size), desc="file_completion_batches")
    for start in pbar:
        end = min(start + batch_size, total)
        batch = items[start:end]
        prompts = []
        for (prompt, target, file_path) in batch:
            trimmed_prompt = tail_trim_prompt(prompt, tokenizer, max_context_tokens)
            prompts.append(trimmed_prompt)
        candidates_per_prompt = [[] for _ in prompts]
        for _ in range(k):
            outs = vllm_client.generate_batch(prompts, max_tokens=min(max_gen_tokens, 2048), temperature=temperature, top_p=top_p)
            for i, out in enumerate(outs):
                candidates_per_prompt[i].append(out)
        for i, (prompt, target, file_path) in enumerate(batch):
            cands = candidates_per_prompt[i]
            best = {"bleu": 0.0, "recall": 0.0, "rouge": 0.0, "em": 0}
            for c in cands:
                rec = token_recall(tokenizer, target, c)
                bl = bleu_score(target, c)
                rl = rouge_l(target, c)
                em = exact_match(target, c)
                best["bleu"] = max(best["bleu"], bl)
                best["recall"] = max(best["recall"], rec)
                best["rouge"] = max(best["rouge"], rl)
                best["em"] = max(best["em"], em)
            records.append({
                "file": file_path,
                "target_len": len(target),
                "prompt_snippet": (prompt[:400] + "...") if len(prompt) > 400 else prompt,
                "target_snippet": (target[:400] + "...") if len(target) > 400 else target,
                "generations": cands,
                "best_bleu": best["bleu"],
                "best_recall": best["recall"],
                "best_rougeL": best["rouge"],
                "exact_match_any": int(best["em"]),
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(out_dir / "file_completion_results.csv", index=False)
        (out_dir / "file_completion_results.json").write_text(df.to_json(orient="records"))
    else:
        (out_dir / "file_completion_results.csv").write_text("")
        (out_dir / "file_completion_results.json").write_text("[]")
    (out_dir / "file_completion_outputs.json").write_text(json.dumps(records, indent=2))
    return df


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, required=True, help="Local path or HF dataset id")
    parser.add_argument("--model_path", type=str, required=True, help="vLLM model path or HF model id")
    parser.add_argument("--out_dir", type=str, default="./recall_out")
    parser.add_argument("--k", type=int, default=3, help="number of completions per prompt")
    parser.add_argument("--batch_size", type=int, default=8, help="prompts per batch when calling vLLM")
    parser.add_argument("--max_files", type=int, default=200, help="limit number of files to evaluate (after filtering)")
    parser.add_argument("--ext_filter", type=str, default=".rs,.res", help="comma-separated file extensions to keep (e.g. .rs,.res)")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max_ref_len", type=int, default=2048, help="max reference length (tokens) to use for scoring")
    parser.add_argument("--max_gen_tokens", type=int, default=2048, help="max tokens to ask model to generate per prompt")
    parser.add_argument("--shuffle", action="store_true", help="shuffle files before truncating to max_files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load files
    files = []
    if is_hf_dataset_id(args.repo_dir):
        try:
            print(f"Loading HF dataset {args.repo_dir} ...")
            files = load_files_from_hf_dataset(args.repo_dir, split="train")
        except Exception as e:
            print("Failed to load HF dataset (will try local dir). Error:", e)
            if Path(args.repo_dir).exists():
                files = load_files_from_local_dir(args.repo_dir)
            else:
                raise
    else:
        if Path(args.repo_dir).exists():
            files = load_files_from_local_dir(args.repo_dir)
        else:
            try:
                files = load_files_from_hf_dataset(args.repo_dir, split="train")
            except Exception as e:
                raise RuntimeError(f"Could not load dataset or local dir from {args.repo_dir}: {e}")

    print(f"Loaded {len(files)} rows/files total from source")

    # apply extension filter
    keep_exts = {e.strip() for e in args.ext_filter.split(",") if e.strip()}
    if keep_exts:
        files = [f for f in files if (f.get("ext") in keep_exts)]
    if args.shuffle:
        random.shuffle(files)
    # limit
    if args.max_files and len(files) > args.max_files:
        files = files[: args.max_files]
    print(f"Filtered -> {len(files)} files (extensions={keep_exts}, max_files={args.max_files})")

    # tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_path, use_fast=True)
    except Exception:
        class SimpleTok:
            def encode(self, x): return x.split()
            def decode(self, toks): return " ".join(map(str, toks))
        tokenizer = SimpleTok()

    # init vllm client
    vllm_client = VLLMWrapper(args.model_path, device=args.device)
    print("Detected model max seq len:", vllm_client.model_max_seq_len)

    # run masked & file completion (batched) with auto-trim prompts & truncated refs
    df_mask = eval_masked_completion_batched(files, vllm_client, tokenizer, out_dir,
                                            k=args.k, batch_size=args.batch_size,
                                            top_p=args.top_p, temperature=args.temperature,
                                            max_gen_tokens=min(args.max_gen_tokens, 1024))
    if df_mask.empty:
        print("No masked-completion records (no functions found on filtered set).")
    else:
        print("Masked completion sample:")
        print(df_mask.head().to_string())

    df_file = eval_file_completion_batched(files, vllm_client, tokenizer, out_dir,
                                          k=args.k, batch_size=args.batch_size,
                                          top_p=args.top_p, temperature=args.temperature,
                                          max_gen_tokens=args.max_gen_tokens, max_ref_len=args.max_ref_len)
    if df_file.empty:
        print("No file-completion records (no sufficiently long files).")
    else:
        print("File completion sample:")
        print(df_file.head().to_string())

    agg = {
        "masked_token_recall_mean": float(df_mask["best_token_recall"].mean()) if (not df_mask.empty and "best_token_recall" in df_mask.columns) else 0.0,
        "masked_exact_match_rate": float(df_mask["exact_match_any"].mean()) if (not df_mask.empty and "exact_match_any" in df_mask.columns) else 0.0,
        "file_bleu_mean": float(df_file["best_bleu"].mean()) if (not df_file.empty and "best_bleu" in df_file.columns) else 0.0,
        "n_masked_records": int(len(df_mask)) if not df_mask.empty else 0,
        "n_file_records": int(len(df_file)) if not df_file.empty else 0,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print("Aggregate metrics saved to", out_dir / "aggregate.json")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
