Our method consists of two steps, where the first is to conduct monolingual summarization, and the second is to conduct crosslingual summarization.

For monolingual summarization, we use [UniSumm](https://github.com/microsoft/UniSumm), which is further fine-tuned on the few-shot monolingual in ConvSumX.

For crosslingual summarization, our method first involves a data-preprocessing step. The preprocessing is to replace original dialogue with one that is concated with monolingual summary, as ```en_summ [lsep] en_doc```, where ```en_summ``` is the english (monolingual) summary, ```en_doc``` is the english dialogue, and ```[lsep]``` is a special token. Then, you can train the model using below command (we show a):

```
python mbart-ft.py \
        --model_path facebook/mbart-large-50-many-to-many-mmt \
        --tokenizer_path facebook/mbart-large-50-many-to-many-mmt \
        --data_root $data_path \
        --src_lang en_XX  --tgt_lang  fr_XX\
        --input_type "doc_en"  --output_type "sum_fr" \
        --save_prefix $prefix \
        --fp32 --device_id 1 \
        --batch_size 8 --epochs 20 \
        --lr $lr --grad_accum $acc --warmup $warm --dataset_size 157 \
        --val_batch_size 16 \
        --min_output_len 5 --max_output_len 50
```

The above is adapted from https://github.com/krystalan/ClidSum/, contributed by [@Huajian Zhang](https://github.com/HJZnlp).
