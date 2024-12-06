import random
import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import DataLoader
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
                 **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, data_collator=data_collator,
                         **kwargs)
        self.processing_class = processing_class
        self.device = device
        self.total_steps = args.num_train_epochs * (len(train_dataset) // args.per_device_train_batch_size)
        self.phase_change_step = self.total_steps // 3
        self.phase2_end_step = 2 * (self.total_steps // 3)
        self.current_step = 0
        self.phase = 'phase1'  # phase1, phase2, phase3
        self.dream_block_size = 100
        self.normal_steps_in_block = 90
        self.dream_steps_in_block = 10
        self.block_step_counter = 0
        self.buffer = []  # Buffer to store data from the last 90 steps

    def training_step(self, model, inputs, *args, **kwargs):
        # Determine current phase
        if self.current_step < self.phase_change_step:
            self.phase = 'phase1'
        elif self.current_step < self.phase2_end_step:
            self.phase = 'phase2'
        else:
            self.phase = 'phase3'

        # if self.phase == 'phase2':
        #
        # else:
        #     # Normal training in phase1 and phase3
        loss = super().training_step(model, inputs)
        return loss