import random
import torch
from transformers import Trainer
from copy import deepcopy

class DreamTrainer(Trainer):
    def __init__(self,
                 model,
                 args,
                 train_dataset=None,
                 eval_dataset=None,
                 data_collator=None,
                 processing_class=None,
                 device='cuda',
                 generation_prompt="Hello world",
                 max_new_tokens=50,
                 max_length=2048,
                 batch_size=8,
                 # Configurable dream parameters
                 dream_block_size=100,
                 normal_steps_in_block=90,
                 dream_steps_in_block=10,
                 num_reorganized_samples=5,
                 num_generated_samples=5,
                 **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, data_collator=data_collator,
                         **kwargs)
        self.processing_class = processing_class
        self.device = device

        # Total steps
        self.total_steps = args.num_train_epochs * (len(train_dataset) // args.per_device_train_batch_size)

        # Phase boundaries
        self.phase_change_step = self.total_steps // 3
        self.phase2_end_step = 2 * (self.total_steps // 3)
        self.current_step = 0

        # Phases: phase1 (normal), phase2 (dream), phase3 (normal)
        self.phase = 'phase1'

        # Dream block parameters (configurable)
        self.dream_block_size = dream_block_size
        self.normal_steps_in_block = normal_steps_in_block
        self.dream_steps_in_block = dream_steps_in_block

        # Check that dream steps match the sum of reorganized and generated
        if num_reorganized_samples + num_generated_samples != self.dream_steps_in_block:
            raise ValueError("num_reorganized_samples + num_generated_samples must equal dream_steps_in_block.")

        self.num_reorganized_samples = num_reorganized_samples
        self.num_generated_samples = num_generated_samples

        self.block_step_counter = 0

        # Buffer to store the last normal_steps_in_block steps of normal training data
        self.buffer = []

        # Will hold the dream data batches for the dream steps
        self.dream_data = []
        self.dream_index = 0
        self.batch_size = batch_size

        # Parameters for generation (if model supports generate)
        self.generation_prompt = generation_prompt
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

        self.model.generation_config.pad_token_id = self.processing_class.pad_token_id

    def reorganize_data(self, buffer, num_samples=5):
        # Simple approach: randomly sample from the buffer
        if len(buffer) < num_samples:
            samples = buffer
        else:
            samples = random.sample(buffer, num_samples)
        return [deepcopy(s) for s in samples]

    def generate_dream_data(self, model, num_samples=5):
        dream_samples = []

        # 先对单一的 prompt 进行 tokenize
        inputs = self.processing_class(
            self.generation_prompt,
            return_tensors='pt'
        ).to(self.device)

        # 每次循环生成 batch_size 个样本，共执行 num_samples 次
        for _ in range(num_samples):
            # 将 prompt 扩展为 batch_size 行（即重复同样的 prompt）
            batch_inputs = {
                key: val.repeat(self.batch_size, 1) for key, val in inputs.items()
            }

            # 一次性生成 batch_size 个结果
            generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True
            )

            # 逐条 decode
            generated_texts = [
                self.processing_class.decode(gid, skip_special_tokens=True)
                for gid in generated_ids
            ]

            # 对 batch_size 个生成文本重新 tokenize，并 pad 到 max_length
            tokenized = self.processing_class(
                generated_texts,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=self.max_length
            ).to(self.device)

            # 构造 batch 数据字典：其中包含 batch_size 行的 input_ids、attention_mask、labels
            batch_sample = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': tokenized['input_ids'].clone()
            }

            dream_samples.append(batch_sample)

        return dream_samples

    def training_step(self, model, inputs, *args, **kwargs):
        # Determine current phase
        if self.current_step < self.phase_change_step:
            self.phase = 'phase1'
        elif self.current_step < self.phase2_end_step:
            self.phase = 'phase2'
        else:
            self.phase = 'phase3'

        # Phase 1 and Phase 3: Normal training
        if self.phase == 'phase1' or self.phase == 'phase3':
            loss = super().training_step(model, inputs, *args, **kwargs)
            self.current_step += 1
            self.block_step_counter = (self.block_step_counter + 1) % self.dream_block_size
            return loss

        # Phase 2: Dream training
        else:
            if self.block_step_counter < self.normal_steps_in_block:
                # Normal step: store inputs in buffer and train normally
                self.buffer.append(inputs)
                loss = super().training_step(model, inputs, *args, **kwargs)
            else:
                # Dream steps
                if self.block_step_counter == self.normal_steps_in_block:
                    # Prepare dream data at start of dream section
                    reorganized_data = self.reorganize_data(self.buffer, num_samples=self.num_reorganized_samples)
                    generated_data = self.generate_dream_data(model, num_samples=self.num_generated_samples)

                    # Combine and shuffle dream data
                    self.dream_data = reorganized_data + generated_data
                    random.shuffle(self.dream_data)
                    self.dream_index = 0

                # Get one dream input batch
                dream_input = self.dream_data[self.dream_index]

                # print(dream_input['input_ids'].shape)

                loss = super().training_step(model, dream_input, *args, **kwargs)
                self.dream_index += 1

                # If finished all dream steps, reset for next block
                if self.dream_index == self.dream_steps_in_block:
                    self.buffer = []
                    self.dream_data = []
                    self.dream_index = 0

            self.current_step += 1
            self.block_step_counter = (self.block_step_counter + 1) % self.dream_block_size
            return loss
