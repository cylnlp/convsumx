import os
import argparse
import random
import numpy as np
from datasets import load_metric,Dataset,DatasetDict
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
import copy
import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50TokenizerFast

import json
from rouge_score import rouge_scorer

def poststr(str_list):
    str_list=[sent.strip() for sent in str_list]
    text="\n".join(str_list)
    return text

def transform_single_dialogsumm_file(file):
    data = open(file,"r").readlines()
    result =[]
    for i in data:
        d = json.loads(i)
        result.append(d)
    return result

class SummarizationDataset(Dataset):
    def __init__(self, split_name, tokenizer, max_input_len, max_output_len, tgt_lang, data_root, model_name,trans,input_type,output_type):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tgt_lang = tgt_lang
        self.model_name = model_name
        self.trans=trans
        self.input_type=input_type
        self.output_type=output_type
        assert tgt_lang in [ 'zh_CN',"en_XX","fr_XX","uk_UA","de_DE"], "Target language in is either Ch, Fr, Uk or En. Please use the correct language identifier."
    
        with open('%s/%s.json'%(data_root,split_name), 'r', encoding='utf-8') as f:
            self.xlds_dataset = json.load(f)
        # self.xlds_dataset = transform_single_dialogsumm_file('%s/%s.json'%(data_root,split_name))

    def __len__(self):
        return len(self.xlds_dataset)

    def __getitem__(self, idx):
        entry = self.xlds_dataset[idx]
        input_ids = self.tokenizer.encode(entry[self.input_type].lower(), truncation=True, max_length=self.max_input_len)
        
        with self.tokenizer.as_target_tokenizer():
            output_ids = self.tokenizer.encode(entry[self.output_type].lower(), truncation=True, max_length=self.max_output_len)
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids


class Summarizer(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams = params

        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.args.tokenizer_path, src_lang=self.args.src_lang, tgt_lang=self.args.tgt_lang)
        self.model = MBartForConditionalGeneration.from_pretrained(self.args.model_path)
        
        #  add speical token and reshape all token embedding
        self.tokenizer.add_tokens(["[lsep]"])
        self.model.resize_token_embeddings(len(self.tokenizer))

        # we once attempt to add speical token act as prompt, such as [summarize], [plan] but no improvement found
        # self.tokenizer.add_tokens(['[summarize]',"[plan]","[lsep]"])

        # we attempt to add speical token like #person1# which is found in our dataset but is not common in pretrained data, however result is even worse
        # self.tokenizer.add_tokens(["[lsep]","#person1#","#person2#","#person3#","#person4#"])

        # the following code for continue training 
        # load a ckpt which already trained on xsumsam
        if self.args.resume_ckpt is not None:
            checkpoint=torch.load(self.args.resume_ckpt)
            checkpoint=checkpoint["state_dict"]
            new_ck=copy.deepcopy(checkpoint)
            prefix="model."
            for key in checkpoint.keys():
                if key.startswith(prefix):
                        new_ck[key[len(prefix):]]=checkpoint[key]
                        new_ck.pop(key)
            self.model.load_state_dict(new_ck)
     
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        self.generated_id = 0

        self.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.args.tgt_lang]
        self.model.config.decoder_start_token_id = self.decoder_start_token_id

    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        return input_ids, attention_mask

    def forward(self, input_ids, output_ids):
        input_ids, attention_mask = self._prepare_input(input_ids)
        decoder_input_ids = output_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)

        labels = output_ids[:, 1:].clone()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )

        lm_logits = outputs[0]
        # Same behavior as modeling_bart.py, besides ignoring pad_token_id
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        return [loss]

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        input_ids, output_ids = batch
        input_ids, attention_mask = self._prepare_input(input_ids)
        
        # when golden is true, we feed the decoder the golden src language summary
        # to see the upper bounder of planing 
        # pls use generation_utils.py in this setting   
        if self.args.golden:
            generated_ids = self.model.generate(
                input_ids=input_ids,
                tgt_ids=output_ids,
                attention_mask=attention_mask,
                use_cache=True,
                num_beams=5,
                max_length = self.args.max_output_len,
                min_length = self.args.min_output_len,
                decoder_start_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang],
                # repetition_penalty=0.6
            )
        else:
            # set repetition_penalty to 0.6 but worse results found
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                num_beams=5,
                max_length = self.args.max_output_len,
                min_length = self.args.min_output_len,
                decoder_start_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang],
                # repetition_penalty=0.6
            )

        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        
        return {'vloss': vloss, 'generated': generated_str,"gold":gold_str}

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        generated_str=[]
        gold_str=[]
    
        for item in outputs:
            generated_str.extend(item['generated'])
            gold_str.extend(item['gold'])
        
        if args.epoch_id is not None:
            self.generated_id=args.epoch_id+1
        print(self.generated_id)
        split_sp_name=""
        if args.val:
            split_sp_name="test"
        else:
            split_sp_name="test"
        with open(self.args.save_dir + '/' + self.args.save_prefix + '/generated_summary_'+split_sp_name+'_%d.txt'%self.generated_id, 'w', encoding='utf-8') as f:
            for ending in generated_str:
                f.write(str(ending)+'\n')
        
        # the following code is to calculate rouge score f1
        rouge_batch1=[]
        rouge_batch2=[]
        rouge_batchL=[]
        for gen,gold in zip(generated_str,gold_str):

            rouge_batch1.append( list(scorer.score(gold,gen)["rouge1"])[2])
            rouge_batch2.append( list(scorer.score(gold,gen)["rouge2"])[2])
            rouge_batchL.append( list(scorer.score(gold,gen)["rougeL"])[2])
        rouge1=sum(rouge_batch1)/len(generated_str)
        rouge2=sum(rouge_batch2)/len(generated_str)
        rougeL=sum(rouge_batchL)/len(generated_str)
        
        rouge1=torch.tensor(rouge1,dtype=torch.float64)
        rouge2=torch.tensor(rouge2,dtype=torch.float64)
        rougeL=torch.tensor(rougeL,dtype=torch.float64)

        self.generated_id += 1
        print(rouge1)
        print(rouge2)
        print(rougeL)

        names = ["vloss"]
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        rouge_name=["vloss","rouge1","rouge2","rougeL"]
        metrics.extend([rouge1,rouge2,rougeL])
        logs = dict(zip(*[rouge_name, metrics]))

        # rouge2 f1 is used to selelct best ckpt in val
        return {'rouge2': rouge2, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)


    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(self.model.parameters(), lr=self.args.lr, scale_parameter=False, relative_step=False)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        num_gpus = 1
        num_steps = self.args.dataset_size * self.args.epochs / num_gpus / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train, tgt_lang, data_root, model_name):
        if current_dataloader is not None:
            return current_dataloader
    
        dataset = SummarizationDataset(split_name = split_name, tokenizer=self.tokenizer, max_input_len=self.args.max_input_len, max_output_len=self.args.max_output_len, \
                                                                            tgt_lang=tgt_lang, data_root=data_root, model_name=model_name,trans=self.args.trans, \
                                                                            input_type=self.args.input_type,output_type=self.args.output_type)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
       
        if split_name != 'train':
            return DataLoader(dataset, batch_size=self.args.val_batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)
        else:
            return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

    @pl.data_loader
    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True, tgt_lang = self.args.tgt_lang, data_root=self.args.data_root, model_name = self.args.model_path)
        return self.train_dataloader_object

    @pl.data_loader
    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'val', is_train=False, tgt_lang = self.args.tgt_lang, data_root=self.args.data_root, model_name = self.args.model_path)
        return self.val_dataloader_object

    @pl.data_loader
    def test_dataloader(self):
        if args.val:
            self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'val', is_train=False, tgt_lang = self.args.tgt_lang, data_root=self.args.data_root, model_name = self.args.model_path)
        else:
            self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False, tgt_lang = self.args.tgt_lang, data_root=self.args.data_root, model_name = self.args.model_path)
        
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):

        ## file root
        parser.add_argument("--save_dir", type=str, default='model_output')
        parser.add_argument("--save_prefix", type=str, default='test')
        parser.add_argument("--model_path", type=str, default='facebook/mbart-large-50-many-to-many-m')
        parser.add_argument("--tokenizer_path", type=str, default='facebook/mbart-large-50-many-to-many-mmt')
        parser.add_argument("--data_root", type=str, default='data/XMediaSum40k')

        ## source language and target language
        parser.add_argument("--src_lang", type=str, default='en_XX')
        parser.add_argument("--tgt_lang", type=str, default='zh_CN')

        ## training details
        parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
        parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--device_id", type=int, default=0, help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=500, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=5e-6, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--max_output_len", type=int, default=128)
        parser.add_argument("--min_output_len", type=int, default=0)
        parser.add_argument("--max_input_len", type=int, default=1024)
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--val", action='store_true', help="val")
        parser.add_argument("--no_progress_bar", action='store_true', help="no progress bar. Good for printing")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")
        parser.add_argument("--epoch_id", type=int, help="defind the id of epoch")
        parser.add_argument("--penelty", action='store_true', help="0.6 ")
        parser.add_argument("--golden", action='store_true', help="generation given planing ")
        parser.add_argument("--dataset_size", type=int, help="num of train set ")
        parser.add_argument("--load_ckpt", type=str, help="Path of a checkpoint to resume from")
        # parser.add_argument("--trans", action='store_true')
        parser.add_argument("--input_type", type=str, help="")
        parser.add_argument("--output_type", type=str, help="")

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = Summarizer(args)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    # point the monitor is rouge2
    # use rouge2 to select ckpt
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=1,
        verbose=True,
        monitor='rouge2',
        mode='max',
        period=-1,
        prefix=''
    )

    print(args)
    print(args.epoch_id)

    # set up the scorer for multilingual rouge 
    global scorer
    if args.tgt_lang=="zh_CN":
        scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],  lang="chinese")
    elif args.tgt_lang=="en_XX":
        scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],  lang="english", use_stemmer=True)
    elif args.tgt_lang=="fr_XX":
        scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],  lang="french", use_stemmer=True)
    elif args.tgt_lang=="uk_UA":
        scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],lang="english", use_stemmer=False)
    elif args.tgt_lang=="de_DE":
        scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],lang="german", use_stemmer=True)

    
    # args.dataset_size = 20000 # the number of training samples

    trainer = pl.Trainer(
        gpus = [args.device_id],
        distributed_backend = 'ddp' if torch.cuda.is_available() else None,
        track_grad_norm = -1,
        max_epochs = args.epochs,
        replace_sampler_ddp = False,
        accumulate_grad_batches = args.grad_accum,
        val_check_interval = args.val_every,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        logger=logger,
        checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
        show_progress_bar=not args.no_progress_bar,
        use_amp=not args.fp32, amp_level='O2',
        resume_from_checkpoint=args.load_ckpt,
    )
    if not args.test:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = Summarizer.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
