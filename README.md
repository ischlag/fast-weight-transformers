# Linear Transformers Are Secretly Fast Weight Memory Systems
This repository contains the code accompanying the paper [*Linear Transformers Are Secretly Fast Weight Memory Systems*](https://arxiv.org/abs/2102.11174) which is currently under review.
It also contains the logs of all synthetic experiments.
## Synthetic Experiments

### Requirements
```bash
$ cat req.txt 
jupyter==1.0.0
pandas==1.0.1
seaborn==0.10.0
torch==1.6.0
matplotlib==3.1.3
numpy==1.17.2
```

```bash
pip3 install -r req.txt
```

### Rerun Experiments
Logs are provided in the ```synthetic/logs``` folder. 
The files in that folder are a result of running the following commands:

Setting 1 (capacity):
```bash
python3 main.py --begin=20 --end=600 --step=20 --attn_name=softmax --update_rule=sum
python3 main.py --begin=20 --end=600 --step=20 --attn_name=linear --update_rule=sum
python3 main.py --begin=20 --end=600 --step=20 --attn_name=dpfp --attn_arg=1 --update_rule=sum
python3 main.py --begin=20 --end=600 --step=20 --attn_name=dpfp --attn_arg=2 --update_rule=sum

python3 main.py --begin=20 --end=600 --step=20 --attn_name=dpfp --attn_arg=3 --update_rule=sum
python3 main.py --begin=20 --end=600 --step=20 --attn_name=favor --attn_arg=64 --update_rule=sum
python3 main.py --begin=20 --end=600 --step=20 --attn_name=favor --attn_arg=128 --update_rule=sum
python3 main.py --begin=20 --end=600 --step=20 --attn_name=favor --attn_arg=512 --update_rule=sum
```

Setting 2 (update rule):
```bash
python3 main.py --begin=20 --end=200 --step=20 --attn_name=dpfp --attn_arg=1 --update_rule=sum --replace
python3 main.py --begin=20 --end=200 --step=20 --attn_name=dpfp --attn_arg=1 --update_rule=ours --replace
python3 main.py --begin=20 --end=200 --step=20 --attn_name=tanh --update_rule=fwm --replace
python3 main.py --begin=20 --end=200 --step=20 --attn_name=dpfp --attn_arg=1 --update_rule=fwm --replace

python3 main.py --begin=20 --end=200 --step=20 --attn_name=dpfp --attn_arg=2 --update_rule=ours --replace
python3 main.py --begin=20 --end=200 --step=20 --attn_name=linear --update_rule=ours --replace
python3 main.py --begin=20 --end=200 --step=20 --attn_name=favor --attn_arg=64 --update_rule=ours --replace
python3 main.py --begin=20 --end=200 --step=20 --attn_name=favor --attn_arg=128 --update_rule=ours --replace
```

Generate figures from the logs using the following notebooks:
```
synthetic/setting1_generate_figure.ipynb
synthetic/setting2_generate_figure.ipynb
```

## Language Modelling & Machine Translation
The toolkit and scripts for language modeling experiments can be found at [IDSIA/lmtool-fwms](https://github.com/IDSIA/lmtool-fwms).

For machine translation experiments, we ported the different attention functions implemented in the language modeling toolkit to the multi-head attention implementation in [FAIRSEQ](https://github.com/pytorch/fairseq).

## Citation
```
@article{schlag2021linear,
      title={Linear Transformers Are Secretly Fast Weight Memory Systems}, 
      author={Imanol Schlag and Kazuki Irie and Jürgen Schmidhuber},  
      journal={Preprint arXiv:2102.11174},
      year={2021}
}
```