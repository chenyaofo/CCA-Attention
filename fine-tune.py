# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['PATH'] += ':/opt/conda/envs/llama31/bin'
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from codebase.core.cca_attn import replace_attn_forward
from torch.distributed import barrier
from external.pretraining import pretraining_slimpajamar_dataset, collate_truncate, pretraining_slimpajamar_llama3
from external.evaluation import evaluation_by_pg19
from torch4x import create_code_snapshot, set_reproducible

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

set_reproducible(24)

@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
	model_type: Optional[str] = field(default="llama")
	rope_theta: Optional[float] = field(default=10000.0)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
	data_dir: Optional[str] = field(default=None)
	optim: str = field(default="adamw_torch")
	model_max_length: int = field(
		default=8192 * 4,
		metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
	)
	replace: bool = field(
		default=False,
		metadata={"help": "Whether replace the attn for training."},
	)
	seed: int = field(default=24, metadata={"help": "random seed for initialization"})
	window_size: int = field(default=1024, metadata={"help": "window size for cca attn."})
	pool_size: int = field(default=16, metadata={"help": "pool size for cca attn."})
	pool_func: str = field(default="cca", metadata={"help": "pool function for cca attn."})
	only_attn: bool = field(default=False, metadata={"help": "Whether only training attention parameters."})
	resume_from_checkpoint: str = field(default=None, metadata={"help": "resume training from checkpoint"})
	
def smart_tokenizer_and_embedding_resize(
	special_tokens_dict: Dict,
	tokenizer: transformers.PreTrainedTokenizer,
	model: transformers.PreTrainedModel,
):
	"""Resize tokenizer and embedding.

	Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
	"""
	num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
	model.resize_token_embeddings(len(tokenizer))

	if num_new_tokens > 0:
		input_embeddings = model.get_input_embeddings().weight.data
		output_embeddings = model.get_output_embeddings().weight.data

		input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
		output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

		input_embeddings[-num_new_tokens:] = input_embeddings_avg
		output_embeddings[-num_new_tokens:] = output_embeddings_avg

def tokenize_fn(tokenizer, example):
	context_length = tokenizer.model_max_length
	outputs = tokenizer(
		tokenizer.eos_token.join(example["text"]),
		truncation=False,
		return_tensors="pt",
		pad_to_multiple_of=context_length,
		padding=True,
	)
	return {"input_ids": outputs["input_ids"].view(-1, context_length)}

def train():
	parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
	model_args, training_args = parser.parse_args_into_dataclasses()
	
	create_code_snapshot(name="code", include_suffix=[".py", ".conf", '.sh'],
						 source_directory="../", store_directory=training_args.output_dir)
	# NOTE: May expand supported model types in the future
	if training_args.replace:
		replace_attn_forward(training_args.pool_size, w_size=training_args.window_size, 
					pool_func=training_args.pool_func)

	# Set RoPE scaling factor
	config = transformers.AutoConfig.from_pretrained(
		model_args.model_name_or_path
	)

	orig_rope_scaling = getattr(config, "rope_scaling", None)
	if orig_rope_scaling is None:
		orig_rope_scaling = {"factor": 1}

	orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
	orig_ctx_len = getattr(config, "max_position_embeddings", None)
	if orig_ctx_len:
		orig_ctx_len *= orig_rope_scaling_factor
		if training_args.model_max_length > orig_ctx_len:
			scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
			config.rope_scaling = {"type": "linear", "factor": scaling_factor}
	config.rope_theta = model_args.rope_theta
	print("rope_theta:", config.rope_theta)

	# Load model and tokenizer
	model = transformers.AutoModelForCausalLM.from_pretrained(
		model_args.model_name_or_path,
		config=config,
		torch_dtype=torch.bfloat16,
	)

	tokenizer = transformers.AutoTokenizer.from_pretrained(
		model_args.model_name_or_path,
		model_max_length=training_args.model_max_length,
		# padding_side="right",
		use_fast=True,
	)
	

	rank = int(os.environ.get('RANK', -1))
	if rank > 0:
		barrier()
		
	# dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=training_args.cache_dir)
	# dataset = dataset.map(partial(tokenize_fn,tokenizer),batched=True, num_proc=128, remove_columns=["text", "meta"])

	# replace the dataset with SlimPajama
	if 'Llama-3' in model_args.model_name_or_path:
		dataset = pretraining_slimpajamar_llama3(training_args.data_dir,seed=training_args.seed)
	elif 'llama-2' in model_args.model_name_or_path:
		dataset = pretraining_slimpajamar_dataset(training_args.data_dir, 
												seed=training_args.seed)
	else:
		raise ValueError('Unsupported model name')
	val_dataset = evaluation_by_pg19('/path/to/pg19', 
									 max_token_length=training_args.model_max_length, 
									 batch_size=1, 
									 tokenizer=tokenizer)

	if rank == 0:
		barrier()

	print(dataset)

	model.config.use_cache = False         # required for gradient checkpointing
	model.enable_input_require_grads()     # required for gradient checkpointing
	model.gradient_checkpointing_enable()  # enable gradient checkpointing
	if training_args.only_attn:
		for name, param in model.named_parameters():
			param.requires_grad = False  # 冻结所有参数
		for name, param in model.named_parameters():
			if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
				param.requires_grad = True
	trainer = Trainer(
		model=model, tokenizer=tokenizer, args=training_args,
		train_dataset=dataset,
		eval_dataset=val_dataset,
		data_collator=partial(collate_truncate, max_token_length=training_args.model_max_length)
		)
	checkpoint=None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	trainer.train(resume_from_checkpoint=checkpoint)
	# trainer.save_state()
	# trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
	train()
