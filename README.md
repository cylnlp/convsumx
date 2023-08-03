# Revisiting Cross-Lingual Summarization: A Corpus-based Study and A New Benchmark with Improved Annotation 	

## ConvSumX


ConvSumX is a cross-lingual conversation summarization benchmark, through a annotation schema that explicitly considers source input context.

ConvSumX consists of 2 sub-tasks: *[DialogSumX](https://github.com/cylnlp/convsumx/tree/main/ConvSumX_data/DialogSumX)* and *[QMSumX](https://github.com/cylnlp/convsumx/tree/main/ConvSumX_data/QMSumX)*, with each covering 3 language directions: En2Zh, En2Fr and En2Ukr.

This work is accepted by ACL 2023. You may find the paper [here](https://aclanthology.org/2023.acl-long.519).

## Citation

Please kindly cite our paper as below:
```
@inproceedings{chen-etal-2023-revisiting,
    title = "Revisiting Cross-Lingual Summarization: A Corpus-based Study and A New Benchmark with Improved Annotation",
    author = "Chen, Yulong  and
      Zhang, Huajian  and
      Zhou, Yijie  and
      Bai, Xuefeng  and
      Wang, Yueguan  and
      Zhong, Ming  and
      Yan, Jianhao  and
      Li, Yafu  and
      Li, Judy  and
      Zhu, Xianchao  and
      Zhang, Yue",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.519",
    pages = "9332--9351",
    abstract = "Most existing cross-lingual summarization (CLS) work constructs CLS corpora by simply and directly translating pre-annotated summaries from one language to another, which can contain errors from both summarization and translation processes.To address this issue, we propose ConvSumX, a cross-lingual conversation summarization benchmark, through a new annotation schema that explicitly considers source input context.ConvSumX consists of 2 sub-tasks under different real-world scenarios, with each covering 3 language directions.We conduct thorough analysis on ConvSumX and 3 widely-used manually annotated CLS corpora and empirically find that ConvSumX is more faithful towards input text.Additionally, based on the same intuition, we propose a 2-Step method, which takes both conversation and summary as input to simulate human annotation process.Experimental results show that 2-Step method surpasses strong baselines on ConvSumX under both automatic and human evaluation.Analysis shows that both source input text and summary are crucial for modeling cross-lingual summaries.",
}
```
