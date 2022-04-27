# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import pdb 
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import datasets
from datasets import load_dataset, load_metric

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
)
from transformers import get_linear_schedule_with_warmup, get_constant_schedule
from transformers import glue_compute_metrics as compute_metrics

import opacus
from opacus.layers import DifferentiallyPrivateDistributedDataParallel as DPDDP

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def rewind(pre_weight):

    recover_dict = {}
    name_list = []
    for ii in range(12):
        name_list.append('bert.encoder.layer.'+str(ii)+'.attention.self.query.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.attention.self.key.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.attention.self.value.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.output.dense.weight')
    name_list.append('bert.pooler.dense.weight')

    for key in pre_weight.keys():

        if 'bert' in key:
            if key in name_list:
                new_key = key+'_orig'
            else:
                new_key = key

            recover_dict[new_key] = pre_weight[key]

    return recover_dict


def see_weight_rate(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.query.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.query.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.key.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.key.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.value.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.value.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.output.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].intermediate.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].intermediate.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].output.dense.weight == 0))


    sum_list = sum_list+float(model.bert.pooler.dense.weight.nelement())
    zero_sum = zero_sum+float(torch.sum(model.bert.pooler.dense.weight == 0))
 

    return 100*zero_sum/sum_list


def pruning_model_custom(model, mask_dict):

    parameters_to_prune =[]
    mask_list = []
    for ii in range(12):
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.query)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight_mask'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.key)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight_mask'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.value)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight_mask'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.output.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight_mask'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].intermediate.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight_mask'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].output.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.output.dense.weight_mask'])

    parameters_to_prune.append(model.bert.pooler.dense)
    mask_list.append(mask_dict['bert.pooler.dense.weight_mask'])

    for ii in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list[ii])


def transfer_grad_samples(model):
    for ii in range(12):
        model.bert.encoder.layer[ii].attention.self.query.weight_orig.grad_sample = model.bert.encoder.layer[ii].attention.self.query.weight.grad_sample * model.bert.encoder.layer[ii].attention.self.query.weight_mask
        model.bert.encoder.layer[ii].attention.self.key.weight_orig.grad_sample = model.bert.encoder.layer[ii].attention.self.key.weight.grad_sample * model.bert.encoder.layer[ii].attention.self.key.weight_mask
        model.bert.encoder.layer[ii].attention.self.value.weight_orig.grad_sample = model.bert.encoder.layer[ii].attention.self.value.weight.grad_sample * model.bert.encoder.layer[ii].attention.self.value.weight_mask
        model.bert.encoder.layer[ii].attention.output.dense.weight_orig.grad_sample = model.bert.encoder.layer[ii].attention.output.dense.weight.grad_sample * model.bert.encoder.layer[ii].attention.output.dense.weight_mask
        model.bert.encoder.layer[ii].intermediate.dense.weight_orig.grad_sample = model.bert.encoder.layer[ii].intermediate.dense.weight.grad_sample * model.bert.encoder.layer[ii].intermediate.dense.weight_mask
        model.bert.encoder.layer[ii].output.dense.weight_orig.grad_sample = model.bert.encoder.layer[ii].output.dense.weight.grad_sample * model.bert.encoder.layer[ii].output.dense.weight_mask
    model.bert.pooler.dense.weight_orig.grad_sample = model.bert.pooler.dense.weight.grad_sample * model.bert.pooler.dense.weight_mask


def clear_grad_samples(model):
    for ii in range(12):
        del model.bert.encoder.layer[ii].attention.self.query.weight.grad_sample
        del model.bert.encoder.layer[ii].attention.self.key.weight.grad_sample
        del model.bert.encoder.layer[ii].attention.self.value.weight.grad_sample
        del model.bert.encoder.layer[ii].attention.output.dense.weight.grad_sample
        del model.bert.encoder.layer[ii].intermediate.dense.weight.grad_sample
        del model.bert.encoder.layer[ii].output.dense.weight.grad_sample
    del model.bert.pooler.dense.weight.grad_sample


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.world_size > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_seed_new(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args, train_dataset, eval_dataset, model, tokenizer, privacy_engine):
    """ Train the model """
    record_result = []

    zero_rate = see_weight_rate(model.module)
    record_result.append(zero_rate)

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.per_gpu_train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True, collate_fn=default_data_collator,
        sampler=torch.utils.data.distributed.DistributedSampler(train_dataset, seed=args.seed)
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    privacy_engine.attach(optimizer)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    scheduler = get_constant_schedule(optimizer)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.per_gpu_train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=True,
    )
    set_seed(args)  # Added here for reproductibility
    for epoch in train_iterator:
        train_dataloader.sampler.set_epoch(epoch)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            if "position_ids" not in batch:
                input_ids = batch["input_ids"]
                batch["position_ids"] = torch.arange(
                    input_ids.shape[1], dtype=torch.long, device=input_ids.device
                ).repeat(input_ids.shape[0], 1)

            # batch = tuple(t.to(args.device) for t in batch)
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            outputs = model(**batch)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss.backward()
            if args.mask_dir:
                transfer_grad_samples(model.module)

            # if args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # # if args.gradient_accumulation_steps > 1:
            # #     loss = loss / args.gradient_accumulation_steps

            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            #     if args.mask_dir:
            #         transfer_grad_samples(model)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                if args.mask_dir:
                    clear_grad_samples(model.module)
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    distributed_sync(args)
                    n_correct_preds, n_samples = evaluate(args, eval_dataset, model, tokenizer)
                    distributed_sync(args)

                    torch.distributed.all_reduce(n_correct_preds, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(n_samples, op=torch.distributed.ReduceOp.SUM)
                    n_correct_preds = n_correct_preds.cpu().numpy()
                    n_samples = n_samples.cpu().numpy()
                    eval_acc = n_correct_preds/n_samples
                    
                    if args.local_rank in [-1, 0]:
                        print(f"eval total number of sample: {n_samples}")

                    results = {}
                    results[args.task_name + "/acc"] = eval_acc

                    if args.local_rank in [-1, 0]:
                        logs = {}

                        record_result.append(results)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        eps, alpha = optimizer.privacy_engine.get_privacy_spent()
                        logs["DP-eps"] = eps
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
            else:
                optimizer.virtual_step()
                if args.mask_dir:
                    clear_grad_samples(model.module)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # results = evaluate(args, model, tokenizer)
    # record_result.append(results)
    # torch.save(record_result, os.path.join(args.output_dir, "result.pt"))

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, tokenizer, prefix=""):

    args.eval_batch_size = args.per_gpu_eval_batch_size * args.world_size

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False, collate_fn=default_data_collator,
        sampler=torch.utils.data.distributed.DistributedSampler(eval_dataset, seed=args.seed)
    )

    # Eval!
    if args.local_rank in [-1, 0]:
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        model.eval()
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        #batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # if args.output_mode == "classification":
    #     preds = np.argmax(preds, axis=1)
    # elif args.output_mode == "regression":
    #     preds = np.squeeze(preds)

    result = compute_metrics(args.task_name, preds, out_label_ids)

    acc = result[args.task_name + "/acc"]
    n_samples = preds.shape[0]
    
    return torch.tensor(acc*n_samples).to(args.device), torch.tensor(n_samples).to(args.device)


def prepare_datasets(args, model, raw_datasets, tokenizer, num_labels):
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        raise Exception("Provide a glue task name")

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    is_regression = args.task_name == "stsb"
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if args.max_seq_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_seq_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser()
    add_gpu_params(parser)

    # Required parameters
    parser.add_argument(
        "--dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--mask_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the glue task",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--weight_pertub",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--per_sample_max_grad_norm", default=1.0, type=float, help="DP per sample max grad norm.")
    parser.add_argument("--target_epsilon", default=0.0, type=float, help="DP target epsilon.")
    parser.add_argument("--noise_multiplier", default=0.0, type=float, help="DP noise multiplier.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()
    parse_gpu(args)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, world_size: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.world_size,
        bool(args.local_rank != -1),
        bool(False),
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    raw_datasets = load_dataset("glue", args.task_name, cache_dir=args.cache_dir)
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.dir == 'pre':
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
    elif args.dir == 'rand':
        model = AutoModelForSequenceClassification.from_config(config=config)

    model.to(args.device)

    if args.weight_pertub:
        load_weight = torch.load(args.weight_pertub, map_location=args.device)
        model_dict = model.state_dict()
        model_dict.update(load_weight)
        model.load_state_dict(model_dict)

    if args.mask_dir:        
        mask = torch.load(args.mask_dir, map_location=args.device)
        pruning_model_custom(model, mask)
        zero_rate = see_weight_rate(model)
        print('model 0:',zero_rate)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    train_dataset, eval_dataset = prepare_datasets(args, model, raw_datasets, tokenizer, num_labels)

    # for n, p in model.named_parameters():
    #     if n.__contains__("weight_orig"):
    #         p.requires_grad = False

    model.train()
    if args.local_rank in [-1, 0]:
        logger.info(f"Number of total parameters: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Number of trainable parameters: {model.num_parameters(only_trainable=True)}")

    model = DPDDP(model)

    if args.target_epsilon > 0.0 and args.noise_multiplier == 0.0:
        privacy_engine = opacus.PrivacyEngine(module=model,
            batch_size=args.per_gpu_train_batch_size*args.gradient_accumulation_steps, sample_size=len(train_dataset),
            max_grad_norm=args.per_sample_max_grad_norm, epochs=args.num_train_epochs,
            target_epsilon=args.target_epsilon, target_delta=1.0/len(train_dataset)
        )
    elif args.target_epsilon == 0.0:
        privacy_engine = opacus.PrivacyEngine(module=model,
            batch_size=args.per_gpu_train_batch_size*args.gradient_accumulation_steps, sample_size=len(train_dataset),
            max_grad_norm=args.per_sample_max_grad_norm, epochs=args.num_train_epochs, noise_multiplier=args.noise_multiplier,
            target_delta=1.0/len(train_dataset)
        )
    else:
        raise Exception("something is wrong with target_epsilon/noise_multiplier")

    if args.local_rank in [-1, 0]:
        logger.info("privacy engine number of replicas: ", privacy_engine.n_replicas)
        logger.info(f"Noise multiplier is: {privacy_engine.noise_multiplier}")

        logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, privacy_engine)
        if args.local_rank in [-1, 0]:
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)


if __name__ == "__main__":
    main()
