from abc import ABC, abstractmethod
import re
from collections import defaultdict
import requests
import json
import numpy as np
from difflib import SequenceMatcher

class QualityCheck(ABC):
    @abstractmethod
    def check(self, s: str) -> bool:
        pass


class ConsecutiveRepeatedSubstringCheck(QualityCheck):
    def __init__(self, minimum_token_length: int = 5, minimum_repeated_times: int = 3):
        self.minimum_token_length = minimum_token_length
        self.minimum_repeated_times = minimum_repeated_times
        self.repeated_substring = None
        self.repeated_substring_token_count = None
        self.repeated_count = None

    @staticmethod
    def _kasai(s, sa):
        n = len(sa)
        rank = [0] * n
        for i in range(n):
            rank[sa[i]] = i
        lcp = [0] * n
        k = 0
        for i in range(n):
            if rank[i] == n - 1:
                k = 0
                continue
            j = sa[rank[i] + 1]
            while j + k < n and i + k < n and s[i + k] == s[j + k]:
                k += 1
            lcp[rank[i]] = k
            k = max(0, k - 1)
        return lcp

    @staticmethod
    def _manber_myers(s, buckets, order=1):
        d = defaultdict(list)
        for bucket in buckets:
            d[s[bucket : bucket + order]].append(bucket)

        res = []
        for _, v in sorted(d.items()):
            if len(v) > 1:
                res.extend(
                    ConsecutiveRepeatedSubstringCheck._manber_myers(s, v, order * 2)
                )
            else:
                res.append(v[0])
        return res

    @staticmethod
    def _longestDupSubstring(s: str) -> str:
        sa = ConsecutiveRepeatedSubstringCheck._manber_myers(s, range(len(s)), 1)
        lcp = ConsecutiveRepeatedSubstringCheck._kasai(s, sa)
        if not any(lcp):
            return ""
        # find the common substrings from longest to shortest
        pos_list = []
        for pos, length in enumerate(lcp):
            # don't consider very short substrings
            if length >= 20:
                substring = s[sa[pos] : sa[pos] + length]
                # check number of tokens
                token_count = len(substring.split())
                if token_count > 1:
                    pos_list.append((token_count, substring))
        pos_list.sort(key=lambda tup: tup[0], reverse=True)     
        return pos_list

    def check(self, s: str) -> bool:
        # find the repeated text which is around minimum token length
        s = re.sub(r' +',' ', s)
        duplicates = ConsecutiveRepeatedSubstringCheck._longestDupSubstring(s)
        # loop through duplicates and check if they are consecutive
        for token_count, repeated_substring in duplicates:
            # find the matches
            matches = []
            for match in re.finditer(re.escape(repeated_substring), s):
                matches.append((match.start(), match.end()))
            # find the consecutive matches
            max_consecutive_repeat = 1
            current_repeat = 1
            for i in range(len(matches) - 1):
                # not exactly consecutive, but close enough
                if matches[i][1] + 100 >= matches[i + 1][0]:
                    current_repeat += 1
                    if current_repeat > max_consecutive_repeat:
                        max_consecutive_repeat = current_repeat
                else:
                    current_repeat = 0
            # check repeated times
            if max_consecutive_repeat < 2:
                continue
            # check for unallowed repeated substrings
            if max_consecutive_repeat * token_count > 20:
                # print(repeated_substring)
                return False
        return True
    
class PromptRepeatCheck(QualityCheck):
    def __init__(self, prompt, maximum_allowed_common_tokens: int = 15):
        self.prompt = prompt
        self.maximum_allowed_common_tokens = maximum_allowed_common_tokens
        self.repeated_substring = None

    def check(self, output: str) -> bool:
        # find the longest common substring
        match = SequenceMatcher(None, self.prompt, output).find_longest_match(
            0, len(self.prompt), 0, len(output)
        )
        self.repeated_substring = self.prompt[match.a : match.a + match.size].strip()
        # check token count
        longest_substring_token_count = len(self.repeated_substring.split())
        if longest_substring_token_count > self.maximum_allowed_common_tokens:
            return False
        return True
    
def consecutive_repeated_substring_metrics(results):
    repeated_metric = []
    for i in range(len(results)):
        check = ConsecutiveRepeatedSubstringCheck().check(results[i])
        if check:
            repeated_metric.append(1)
        else:
            repeated_metric.append(0)
    return sum(repeated_metric) / len(repeated_metric), repeated_metric
    
### SUMMAC
def get_summac_score(contexts, generated, batch_size=25):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer SrH_UhFfpytl3pw7iYmguw:dQBFt5Mo0ywITKEVuGRcb1ZB9jrg4_99QUrPSR8xVSw'
    }
    
    summac_precision_list = []
    for i in range(0, len(generated), batch_size):

        json_data = {
            "generated": generated[i:i+batch_size],
            "reference": contexts[i:i+batch_size],
            "task": contexts[i:i+batch_size],
            "score_type": "summac"
        }

        response = requests.post(
            # 'https://snow-task_intel_org-flash_flock_1.job.console.elementai.com/api',
            # 'https://5c0b6dde-c16e-4050-ac86-80e2a82f15a6-8080.job.console.elementai.com/api',
            'https://216690fb-7067-4474-87b4-367d0b654e2e-8080.job.console.elementai.com/api',
            headers=headers,
            json=json_data,
            verify=False,
        )

        output = response._content.decode('utf-8')
        output_dict = json.loads(output)['score']
        summac_precision_list += output_dict['summac-precision']
        
    json_data = {
        "generated": generated[i:],
        "reference": contexts[i:],
        "task": contexts[i:],
        "score_type": "summac"
    }

    response = requests.post(
        # 'https://snow-task_intel_org-flash_flock_1.job.console.elementai.com/api',
        #'https://5c0b6dde-c16e-4050-ac86-80e2a82f15a6-8080.job.console.elementai.com/api',
        'https://216690fb-7067-4474-87b4-367d0b654e2e-8080.job.console.elementai.com/api',
        headers=headers,
        json=json_data,
        verify=False,
    )

    output = response._content.decode('utf-8')
    output_dict = json.loads(output)['score']
    summac_precision_list += output_dict['summac-precision']
    return {'summac-precision': np.mean(summac_precision_list), 'summac-precision-list': summac_precision_list}

def get_rouge_score(generated, reference):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer SrH_UhFfpytl3pw7iYmguw:dQBFt5Mo0ywITKEVuGRcb1ZB9jrg4_99QUrPSR8xVSw'
    }
        
    json_data = {
        "generated": generated,
        "reference": reference,
        "task": reference,
        "score_type": "rouge"
    }

    response = requests.post(
        # 'https://snow-task_intel_org-flash_flock_1.job.console.elementai.com/api',
        #'https://5c0b6dde-c16e-4050-ac86-80e2a82f15a6-8080.job.console.elementai.com/api',
        'https://216690fb-7067-4474-87b4-367d0b654e2e-8080.job.console.elementai.com/api',
        headers=headers,
        json=json_data,
        verify=False,
    )

    output = response._content.decode('utf-8')
    output_dict = json.loads(output)['score']
    for key, value in output_dict.items():
        output_dict[key] = np.mean(value)
    return output_dict


def format_metrics(labels, results, target_sections):
    section_format_score = []
    for single_result in results:
        single_format_score = 0
        for section in target_sections:
            if len(re.findall(section + r"[:\s]", single_result)) == 1:
                single_format_score += 1
        single_format_score = single_format_score / len(target_sections)
        section_format_score.append(single_format_score)
    return section_format_score


def prompt_repeat_metrics(prompt, results):
    repeatition_metric = []
    for i in range(len(results)):
        check = PromptRepeatCheck(prompt).check(results[i])
        if check:
            repeatition_metric.append(1)
        else:
            repeatition_metric.append(0)
    return repeatition_metric

def entity_metrics(labels, results):
    regex = r"\b(?!\S*\.)\S+\d{3,}\b"
    entity_metric = []
    for i in range(len(results)):
        # get entities in labels
        label_entities = re.findall(regex, labels[i].lower())
        # check if entities in labels are in results
        if len(label_entities) == 0:
            entity_metric.append(1)
            continue
        total_found = 0
        for entity in label_entities:
            if entity in results[i].lower():
                total_found += 1
        entity_metric.append(total_found / len(label_entities))
    return entity_metric


def entity_hallucination_metrics(inputs, labels, results):
    regex = r"\b(?!\S*\.)\S+\d{3,}\b"
    entity_metric = []
    for i in range(len(results)):
        # get entities in results
        result_entities = re.findall(regex, results[i].lower())
        # check if entities in inputs are in results
        if len(result_entities) == 0:
            entity_metric.append(1)
            continue
        score = 1
        for entity in result_entities:
            if entity not in inputs[i].lower():
                score = 0
                break
        entity_metric.append(score)
    return entity_metric
       