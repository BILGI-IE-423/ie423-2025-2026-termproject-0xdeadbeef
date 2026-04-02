# IE 423 Term Project Proposal — Sarcasm Detection in News Headlines

## Team Members
- Ada Güner Nohut 122203012
- Onur Sarıdoğan 121203077
- Veli Erenay Açıl 122203072

## Dataset
**News Headlines Dataset for Sarcasm Detection**
- Source: [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- 28,619 news headlines labeled as sarcastic (The Onion) or not sarcastic (HuffPost)

## Repository Structure
```
├── README.md                  → you are here
├── requirements.txt           → required Python packages
├── data/
│   ├── raw/                   → raw dataset file (see data/README.md)
│   ├── processed/             → cleaned and preprocessed data
│   └── README.md              → dataset access and placement instructions
├── scripts/
│   ├── 01_load_data.py        → loads dataset, checks paths, prints basic info
│   ├── 02_preprocess_data.py  → cleans text, engineers features, saves processed data
│   └── 03_basic_eda.py        → generates 5 visualizations and summary tables
├── outputs/
│   ├── figures/               → generated plots
│   └── tables/                → generated summary tables
└── docs/
    └── ResearchProposalPreprocessing.md   → main proposal and presentation document
```

## Installation
```bash
pip install -r requirements.txt
```

## Running the Code
```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```

## Proposal Document
See: [docs/ResearchProposalPreprocessing.md](docs/ResearchProposalPreprocessing.md)
