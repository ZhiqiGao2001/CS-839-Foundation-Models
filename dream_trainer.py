import random
import torch
from transformers import Trainer
from copy import deepcopy
import math


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

        steps_per_epoch = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
        original_total_steps = args.num_train_epochs * steps_per_epoch

        # 检查梦境步骤配置是否正确
        if num_reorganized_samples + num_generated_samples != dream_steps_in_block:
            raise ValueError("num_reorganized_samples + num_generated_samples must equal dream_steps_in_block.")

        self.num_reorganized_samples = num_reorganized_samples
        self.num_generated_samples = num_generated_samples

        self.dream_block_size = dream_block_size
        self.normal_steps_in_block = normal_steps_in_block
        self.dream_steps_in_block = dream_steps_in_block

        # 将训练分为三段（phase1, phase2, phase3）
        phase1_steps = original_total_steps // 3
        phase2_steps = original_total_steps // 3
        phase3_steps = original_total_steps - phase1_steps - phase2_steps

        # 重新计算 phase2 的实际 steps 数量（加入 dream steps）
        num_full_blocks = phase2_steps // self.normal_steps_in_block
        remainder = phase2_steps % self.normal_steps_in_block
        phase2_new_steps = num_full_blocks * self.dream_block_size + remainder

        new_total_steps = phase1_steps + phase2_new_steps + phase3_steps
        self.total_steps = new_total_steps

        # 同步更新 state 和 args
        self.state.max_steps = new_total_steps
        self.args.max_steps = new_total_steps

        # 更新 phase 的分界点
        self.phase_change_step = phase1_steps
        self.phase2_end_step = phase1_steps + phase2_new_steps

        self.buffer = []
        self.dream_data = []
        self.dream_index = 0
        self.batch_size = batch_size

        self.generation_prompt = generation_prompt
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

        self.model.generation_config.pad_token_id = self.processing_class.pad_token_id

        self.training_step_count = 0  # 自定义步骤计数

    def reorganize_data(self, buffer, num_samples=5):
        if len(buffer) < num_samples:
            samples = buffer
        else:
            samples = random.sample(buffer, num_samples)
        return [deepcopy(s) for s in samples]

    def generate_dream_data(self, model, num_samples=5):
        model.eval()
        dream_samples = []

        inputs = self.processing_class(
            self.generation_prompt,
            return_tensors='pt'
        ).to(self.device)

        for _ in range(num_samples):
            batch_inputs = {key: val.repeat(self.batch_size, 1) for key, val in inputs.items()}

            generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True
            )

            generated_texts = [
                self.processing_class.decode(gid, skip_special_tokens=True)
                for gid in generated_ids
            ]

            tokenized = self.processing_class(
                generated_texts,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=self.max_length
            ).to(self.device)

            batch_sample = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': tokenized['input_ids'].clone()
            }

            dream_samples.append(batch_sample)
        model.train()

        return dream_samples

    def training_step(self, model, inputs, *args, **kwargs):
        # 使用 self.training_step_count 而不是 global_step
        current_step = self.training_step_count

        # 根据 current_step 判断 phase
        if current_step < self.phase_change_step:
            self.phase = 'phase1'
        elif current_step < self.phase2_end_step:
            self.phase = 'phase2'
        else:
            self.phase = 'phase3'

        if self.phase in ['phase1', 'phase3']:
            loss = super().training_step(model, inputs, *args, **kwargs)
        else:
            # phase2 logic
            block_pos = (current_step - self.phase_change_step) % self.dream_block_size

            if block_pos < self.normal_steps_in_block:
                self.buffer.append(inputs)
                loss = super().training_step(model, inputs, *args, **kwargs)
            else:
                if block_pos == self.normal_steps_in_block:
                    reorganized_data = self.reorganize_data(self.buffer, num_samples=self.num_reorganized_samples)
                    generated_data = self.generate_dream_data(model, num_samples=self.num_generated_samples)
                    self.dream_data = reorganized_data + generated_data
                    random.shuffle(self.dream_data)
                    self.dream_index = 0

                # 确保 dream_data 不为空并且 index 不越界
                if self.dream_index >= len(self.dream_data):
                    raise IndexError("dream_index out of range. Possibly logic error in dream steps calculation.")

                dream_input = self.dream_data[self.dream_index]
                loss = super().training_step(model, dream_input, *args, **kwargs)
                self.dream_index += 1

                if self.dream_index == self.dream_steps_in_block:
                    self.buffer = []
                    self.dream_data = []
                    self.dream_index = 0

        # 在最后递增计数器
        self.training_step_count += 1
        return loss
