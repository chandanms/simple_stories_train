import os
import random
import re
from typing import Any

import torch
from datasets import Features, Value, load_dataset, DatasetDict
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Dataset, Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    get_model,
    modelapi,
)
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState, generate
from inspect_ai.tool import ToolChoice, ToolInfo
from models.llama import Llama
from models.model_configs import MODEL_CONFIGS
from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    is_tokenized: bool
    tokenizer_file_path: str
    streaming: bool
    split: str
    n_ctx: int = 1024
    seed: int
    column_name: str
    data_files: str


class CustomStoriesDataset(Dataset):
    def __init__(self, data_files: str, limit: int = 1000, prompt_words: int = 18):
        dataset_config = DatasetConfig(
            name="json",
            is_tokenized=False,
            tokenizer_file_path="tokenizer/tokenizer.json",
            streaming=True,
            split="train",
            n_ctx=512,
            seed=42,
            column_name="text",
            data_files=data_files
        )

        features = Features({
            'text': Value('string')
        })

        self.prompt_words = prompt_words

        self.dataset = load_dataset(
            dataset_config.name,
            streaming=dataset_config.streaming,
            split=dataset_config.split,
            data_files=dataset_config.data_files,
            features=features
        )

        # Convert streaming dataset to list and limit size
        self.samples = []
        for idx, item in enumerate(self.dataset.take(limit)): # type: ignore
            truncated_text = self._truncate_text(item['text'])
            self.samples.append(Sample(input=truncated_text, id=str(idx)))

        self._data_files = data_files
        self._shuffled = False
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to specified number of words."""
        words = text.split()
        if len(words) <= self.prompt_words:
            return text
        return ' '.join(words[:self.prompt_words])

    def __getitem__(self, index: int) -> Sample: # type: ignore
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)
        
    def filter(self, predicate: callable) -> 'CustomStoriesDataset': #type: ignore
        filtered_samples = [sample for sample in self.samples if predicate(sample)]
        new_dataset = CustomStoriesDataset(self._data_files, len(filtered_samples))
        new_dataset.samples = filtered_samples
        return new_dataset

    @property
    def location(self) -> str:
        return self._data_files

    @property
    def name(self) -> str:
        return "custom_stories_dataset"
    
    def shuffle(self, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.samples)
        self._shuffled = True

    def shuffle_choices(self, seed: int | None = None) -> None:
        # No choices to shuffle in this dataset
        pass

    @property
    def shuffled(self) -> bool:
        return self._shuffled

    def sort(self, key: callable) -> None: #type: ignore
        self.samples.sort(key=key)

class CustomStoriesAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )
        
        model_path = model_args.get("model_path")
        tokenizer_path = model_args.get("tokenizer_path")
        device = model_args.get("device", "cuda")
        
        if not model_path or not tokenizer_path:
            raise ValueError("model_path and tokenizer_path are required")
            
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        model_config = MODEL_CONFIGS["33M"]
        self.model = Llama.from_pretrained(model_path, model_config)
        self.model.to(device)
        self.model.eval()
        self.device = device

    async def generate(
        self,
        input: list[ChatMessage] | str,
        tools: list[ToolInfo] | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        # Handle both string and ChatMessage input
        prompt = input[0].content if isinstance(input, list) else input
        
        # Tokenize input
        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(encoding.ids).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        eos_token_id = self.tokenizer.token_to_id("<|endoftext|>") 

        # Set up generation config
        gen_config = config or self.config
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                idx=input_ids,
                max_new_tokens=gen_config.max_tokens if gen_config.max_tokens else 100,
                temperature=gen_config.temperature if gen_config.temperature else 0.7,
                top_k=40,
                eos_token_id=eos_token_id
            )

        # Decode output
        output_text = self.tokenizer.decode(output_ids[0].tolist())

        # Create model output
        choice = ChatCompletionChoice(
            message=ChatMessageAssistant(content=output_text),
            logprobs=None,
        )

        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=ModelUsage(
                input_tokens=len(encoding.ids),
                output_tokens=len(output_ids[0]) - len(encoding.ids),
                total_tokens=len(output_ids[0])
            ),
        )



@scorer(
    metrics={
        "originality": [mean()], 
        "coherence": [mean()], 
        "grammar": [mean()],
        "quality": [mean()]
    }
)
def story_quality_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        load_dotenv()
        api_key = os.getenv("API_KEY")
        judge_model = get_model("openai/gpt-4", api_key=api_key)

        generated_story = state.output.completion
        print (generated_story)

        answer_match = re.search(r"ANSWER:(.*?)(?=ANSWER:|$)", generated_story, re.DOTALL)
        story = answer_match.group(1).strip() if answer_match else generated_story

        # Craft prompt for the judge
        evaluation_prompt = f"""
Evaluate the following story based on four criteria by assigning each a score from 0 to 100:
1. **Originality**: Rate the creativity and uniqueness of the story.
2. **Coherence**: Rate the logical flow and consistency of the story.
3. **Grammar**: Rate the grammatical correctness of the story. Ignore spacing and capitalization.
4. **Quality**: Rate the overall quality of the story.
You should also provide a short explanation for your judgment.

**Story to evaluate:**
{story}

Please provide your assessment in the following format, ensuring each score is an integer between 0 and 100:
{{"EXPLANATION": "The dialogue is coherent, but the phrasing is slightly off.","ORIGINALITY": 0, "COHERENCE": 0, "GRAMMAR": 0, "QUALITY": 0}}
"""

         # Get evaluation from judge model
        result = await judge_model.generate(
            evaluation_prompt, config=GenerateConfig(temperature=0.0)
        )

        # Use regex to find a dictionary with explanation and four scores
        dict_match = re.search(
            r'\{"EXPLANATION":\s*"([^"]+)",\s*"ORIGINALITY":\s*(\d+),\s*"COHERENCE":\s*(\d+),\s*"GRAMMAR":\s*(\d+),\s*"QUALITY":\s*(\d+)\}',
            result.completion,
        )

        if dict_match:
            explanation = dict_match.group(1)
            scores = {
                "originality": int(dict_match.group(2)),
                "coherence": int(dict_match.group(3)),
                "grammar": int(dict_match.group(4)),
                "quality": int(dict_match.group(5)),
            }
        else:
            explanation = ""
            scores = {"originality": 0, "coherence": 0, "grammar": 0, "quality": 0}

        scores = {k: max(0, min(v, 100)) for k, v in scores.items()}

        return Score(
            value=scores,  # type:ignore
            answer=story,
            explanation=explanation,
        )

    return score

@task
def evaluate_story_generation():
    """Task definition for evaluating story generation capabilities."""
    dataset = CustomStoriesDataset(
        data_files="simplestories_processed_data/test.jsonl",
        limit=100  # Adjust as needed
    )
    
    return Task(
        dataset=dataset,
        plan=[generate()],
        scorer=story_quality_scorer(),
    )

@modelapi(name="custom_stories")
def custom_stories():
    return CustomStoriesAPI

if __name__ == "__main__":
    model = get_model(
        "custom_stories/training_outputs/simplestories_60ksteps/checkpoints/model.pt",
        model_path="training_outputs/simplestories_60ksteps/checkpoints/model.pt",
        tokenizer_path="tokenizer/tokenizer.json"
    )
    
    eval(evaluate_story_generation, model=model, limit=100, max_tokens=300)