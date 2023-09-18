import logging
from typing import List
from transformers import AutoTokenizer
from preprocessing.summarization.summarization_preprocessing import (
    RemovePreviousSummary,
    ExportPrompt,
    RemoveHtmlTags,
    RemoveLinks,
    TruncateInput,
)
from preprocessing.common import PreProcessedInput

_logger = logging.getLogger(__name__)

class PreProcessingPipeline:
    preprocessors = {
        "prompt": ExportPrompt(),
        "remove_previous_summary": RemovePreviousSummary(),
        "html_tags": RemoveHtmlTags(),
        "remove_links": RemoveLinks(),
        "truncate_input": TruncateInput(),
    }
    default_steps = ["prompt", "remove_links", "html_tags", "remove_previous_summary", "truncate_input"]

    def preprocess(
        self, input: str, tokenizer: AutoTokenizer, steps: List[str]
    ) -> PreProcessedInput:
        input = PreProcessedInput(input)
        # apply steps one by one
        for step in steps:
            if step not in PreProcessingPipeline.preprocessors:
                continue
            try:
                input = PreProcessingPipeline.preprocessors[step].preprocess(
                    input, tokenizer
                )
            except Exception as exp:
                _logger.error(f"Error in preprocessing step {step}: {exp}")
        return input
