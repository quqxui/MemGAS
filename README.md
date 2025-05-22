# MemGAS

### Environment
    conda create -n memgas python=3.9
    conda activate memgas
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt


### Data Process
Download the original data:

    - Download the LongMemEval-s LongMemEval-m datasets from https://github.com/xiaowu0162/LongMemEval
    - Download the LoCoMo-10 dataset from https://github.com/snap-research/locomo/blob/main/data/locomo10.json
    - Download the Long-MT-Bench+ dataset from https://huggingface.co/datasets/panzs19/Long-MT-Bench-Plus

Put the original data in `data/origin_data/`, and  run:

```python
cd data/
python dataprocess.py
```
The processed data will be available in `data/process_data/`

### Generating the multigranularity information
**Enter your API key and URL** in `src/construct/multigran_generation.py` and `src/generation/async_llm.py`, and run:

```python
cd src/construct/
multigran_generation.py
```
The generated results will be stored in  `../../multi_granularity_logs/`

### Memory Retrieval
Run the following commands to reproduce single granularity and multi-granularity results:

```python
cd src/retrieval/


# single granularity
python3 run_retrieval.py --dataset locomo10 --retriever contriever --method session_level
python3 run_retrieval.py --dataset locomo10 --retriever contriever --method turn_level
python3 run_retrieval.py --dataset locomo10 --retriever contriever --method summary_level
python3 run_retrieval.py --dataset locomo10 --retriever contriever --method key_level

# multi granularity
python3 run_retrieval.py --dataset locomo10 --retriever contriever --method hybrid_level

python3 run_retrieval.py \
  --dataset locomo10 \
  --retriever contriever \
  --method multigran \
  --num_seednodes 15 \
  --mem_threshold 30 \
  --damping 0.1 \
  --temp 0.2

```
Change the --dataset parameter to  `locomo10` `longmemeval_s` `longmemeval_m` `LongMTBench+` for experiments on other datasets.


### Generation
To conduct QA experiements, run:
```python
cd src/generation/

python generation.py --dataset locomo10 --retriever contriever --model_name_or_path gpt-4o-mini --topk 3 --method multigran

```

The generated results will be saved in `generation_logs`, and the metrics can be found in `generation_logs/metrics/`

#### Evaluation
Evaluating with GPT4o:

The code only requires --eval_file for geneated path, --model_name_or_path for evaluation:
```python
cd src/evaluation/
python llm_judge_single.py --model_name_or_path gpt-4o --eval_file  locomo10-contriever-multigran_filter-gpt-4o-mini-topk_3.jsonl

```

Evaluating different query types:
```python
cd src/evaluation/
python eval_query_type.py --eval_file ../../generation_logs/locomo10-contriever-multigran_filter-gpt-4o-mini-topk_3.jsonl
```

