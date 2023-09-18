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
from consec_repeat import consecutive_repeated_substring_metrics, get_summac_score, get_rouge_score, format_metrics, prompt_repeat_metrics, entity_metrics, entity_hallucination_metrics
from transformers import LogitsProcessorList

from custom_logits_processor import (NoRepeatNGramLogitsProcessor,)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def infer(eval_file, prompt_data):
    with open(eval_file, encoding="utf-8") as f:
        eval_data = json.load(f)
    
    # TODO: remove this
    # eval_data = eval_data[:2]

    results_hallucination_v030 = []
    inputs_list_hallucination_v030 = []
    targets_hallucination_v030 = []

    output_list = []
    context = []
    prompts = []
    output = []
    id = 0
    for record in tqdm(eval_data):
        prompt = prompt_data[str(id)]
        curr_output = {}
        inputs = record["inputs_pretokenized"]
        # inputs = record["Prompt"]
        custom_logits_processors = LogitsProcessorList()
        no_repeat_ngram_size = 10
        custom_logits_processors.append(
            NoRepeatNGramLogitsProcessor(no_repeat_ngram_size, tokenizer)
        )

        # preprocessing
        # steps = ["remove_previous_summary", "html_tags", "truncate_input"]
        steps = ["remove_previous_summary", "truncate_input"]
        pipeline = PreProcessingPipeline()

        def preprocess(text):

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
                    logits_processor=custom_logits_processors,
                    num_return_sequences=1,
                    repetition_penalty=1.05
            )

            outputs = outputs[:, inputs_tokenized["input_ids"].shape[1] :]

            single_result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )

            curr_output['input'] = inputs
            curr_output['target'] = record["targets_pretokenized"]
            # curr_output['target'] = record["Edited Response"]
            curr_output['model output'] = single_result[0]
            curr_output['id'] = str(id) #record['id']
            # curr_output['Eval_type'] = record['Eval_type']
            id += 1
            
            context.append(inputs.split('<|end|>\n<|user|>\n')[0].split('<|system|>\n')[1])
            prompts.append(prompt)
            output.append(single_result[0])
            
        output_list.append(curr_output)

    df = pd.DataFrame({'Context': context, 'Prompt': prompts, 'Output': output})
    df.to_excel('context_unique_prompt_output.xlsx')

if __name__ == '__main__':
    model_id = '/mnt/atg_platform/models/now_llm_chat/v0.4.0-rc2'
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
    eval_file = 'eval_data/eval_set_v2.0.json'
    
    prompts = json.load(open('prompts.json'))
    infer(eval_file, prompts)