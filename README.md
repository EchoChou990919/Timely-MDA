# Timely-MDA
Timely-MDA: A Benchmark for Generalizable MiRNA-Disease Association Prediction

## Organization of this repository

```bash
Timely-MDA
  ├─ data
    ├─ associations      # raw data from DisGeNet*, miRTarBase, HMDD, RNADisease, and HumanNet
    ├─ entities          # raw data from MeSH, miRBase, and HGNC
    ├─ RNA-FM            # please get from RNA-FM*
    ├─ PubMedBERT        # please get from PubMedBERT*
    ├─ our_data          # our processed data
    ├─ dataset_construction.ipynb   # codes of the dataset construction
    └─ preprocessing.ipynb          # codes of the data preprocessing, including the data split
  ├─ model_weights       # the trained model weights of PLM-HGNN
  ├─ model.py            # PLM-HGNN
  ├─ utils.py            # functions utilized in the training and evaluation process
  ├─ demo.ipynb          # training and evaluate of PLM-HGNN in a file
  ├─ similarity_utils.py      # functions utilized in calculating miRNA-miRNA / disease-disease similarities
  ├─ requirements.txt    # relevant environmental dependencies
  └─ README.md
```

Attention*:  
- DisGeNet: https://www.disgenet.org/
- RNA-FM: https://github.com/ml4bio/RNA-FM
- PubMedBERT: https://huggingface.co/NeuML/pubmedbert-base-embeddings

## Baselines

Please get the existing baseline methods from their own repositories:
- NIMCGCN: https://github.com/ljatynu/NIMCGCN/
- DFELMDA: https://github.com/ljatynu/NIMCGCN/
- AGAEMD: https://github.com/Zhhuizhe/AGAEMD
- MINIMDA: https://github.com/chengxu123/MINIMDA

Many thanks to the authors for their generous sharing!

## Concat
If you have any questions, welcome to contact Xian Guan at guanxian@stu.scu.edu.cn!
