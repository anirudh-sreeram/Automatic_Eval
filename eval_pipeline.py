import sys
import glob, pandas as pd, numpy as np, re
import transformers
from transformers import pipeline
from tqdm import tqdm
from datasets import load_dataset,concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoConfig
import torch
import warnings
import os, json
warnings.filterwarnings("ignore")
from tqdm import tqdm
from accelerate import init_empty_weights, infer_auto_device_map
import argparse
import gc
import json
from tqdm import tqdm
import os
import argparse
import time
from preprocessing.preprocessing_pipeline import PreProcessingPipeline
from transformers import AutoTokenizer
import datetime
from consec_repeat import consecutive_repeated_substring_metrics, get_summac_score, get_rouge_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def infer(eval_file, output_file_name, prompt):
    with open(eval_file, encoding="utf-8") as f:
        eval_data = json.load(f)
    
    # TODO: remove this
    # eval_data = eval_data[:2]

    results_hallucination_v030 = []
    inputs_list_hallucination_v030 = []
    targets_hallucination_v030 = []

    output_list = []
    for record in tqdm(eval_data):
        curr_output = {}
        inputs = record["inputs_pretokenized"]

        # preprocessing
        steps = ["remove_previous_summary", "html_tags", "truncate_input"]
        pipeline = PreProcessingPipeline()

        def preprocess(text):
            text = (
                text
                    .replace("<|system|>", "<|system|>\n")
                    .replace("<|endoftext|><|customer|>", "<|end|>\n<|user|>\n")
                    .replace("<|endoftext|><|agent|>", "<|end|>\n<|assistant|>")
            )

            return pipeline.preprocess(text, tokenizer, steps).preprocessed_input

        inputs = preprocess(inputs)
        
        # need to add current inputs
        inputs = inputs.split('<|end|>\n<|user|>\n')[0] + '<|end|>\n<|user|>\n' + prompt + '<|end|>\n<|assistant|>'
        
        cuda_device = 'cuda:0'
        inputs_tokenized = tokenizer(inputs, padding=True, return_tensors="pt")

        with torch.no_grad():
            inputs_tokenized = {k: v.to(cuda_device) for k, v in inputs_tokenized.items()}
            outputs = model_base.generate(
                    input_ids=inputs_tokenized["input_ids"],
                    attention_mask=inputs_tokenized["attention_mask"],
                    max_new_tokens=500,
                    temperature=0.3,
                    num_beams=1,
                    use_cache=True,
                    do_sample=True,
                    no_repeat_ngram_size=10,
                    repetition_penalty=1.05,
                    num_return_sequences=1,
            )

            outputs = outputs[:, inputs_tokenized["input_ids"].shape[1] :]

            single_result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )

            curr_output['input'] = inputs
            curr_output['target'] = record["targets_pretokenized"]
            curr_output['model output'] = single_result[0]
            #curr_output['id'] = record['id']
            #curr_output['Eval_type'] = record['Eval_type']
            
        output_list.append(curr_output)

    with open(output_file_name, 'w') as f:
        json.dump(output_list, f, indent=4)
    
def get_metrics(output_file_name):
    with open(output_file_name, 'r') as f:
        data = json.load(f)
    contexts = []
    generated = []
    reference = []
    for sample in data:
        contexts.append(sample['input'].split('<|end|>\n<|user|>\n')[0].strip('<|system|>\n'))
        generated.append(sample['model output'])
        reference.append(sample['target'])
    consecutive_repeated_substring_score, repeated_metric = consecutive_repeated_substring_metrics(generated)
    summac_score = get_summac_score(contexts, generated)
    rouge_score = get_rouge_score(generated, reference)
    return consecutive_repeated_substring_score, repeated_metric, summac_score, rouge_score

def add_individual_scores_to_json(json_file, summac_score, repeated_metric):
    output_data = json.load(open(json_file, 'r'))
    for idx, output_sample in enumerate(output_data):
        output_sample['summac precision'] = summac_score['summac-precision-list'][idx]
        output_sample['repeated'] = repeated_metric[idx]
    output_data = sorted(output_data, key=lambda x: x['summac precision'])
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    # model_id = '/mnt/atg_platform/models/other/nowllm_15.5b_v0.3_082623'
    model_id = '/mnt/atg_platform/models/now_llm_chat/v0.3.0_nowassist' # the one Anil used
    cache = 'cache_model'

    # creating model
    model_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache,
        trust_remote_code=True,
        use_cache=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model_base.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    dt_now = datetime.datetime.now()
    folder_name = '{}_{}_{}'.format(dt_now.month, dt_now.day, dt_now.year)
    path = 'OUTPUTS/{}'.format(folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
        
    # eval_file = 'eval_data/combined_eval_hall.json'
    eval_file = 'eval_data/eval_set_v2.0.json'
    # prompt_data = json.load(open('prompts/prompts_8_30_2023.json', 'r'))
    prompt_data = json.load(open('prompts/prompts_8_30_2023.json', 'r'))
    metric_data = {}
    
    start = time.time()
    for prompt_index in prompt_data:
        output_file_name = os.path.join(path, 'inputs_{}_outputs.json'.format(prompt_index))
        infer(eval_file, output_file_name, prompt_data[prompt_index])
        consecutive_repeated_substring_score, repeated_metric, summac_score, rouge_score = get_metrics(output_file_name)
        metric_data[prompt_index] = {
            'Consecutive repeated substring score': consecutive_repeated_substring_score,
            'Summac precision': summac_score['summac-precision'],
            'Rouge': rouge_score
        }
        add_individual_scores_to_json(output_file_name, summac_score, repeated_metric)
    print('Duration: {}'.format(time.time() - start))
    
    with open(os.path.join(path, 'metrics.json'), 'w') as f:
        json.dump(metric_data, f, indent=4)
    