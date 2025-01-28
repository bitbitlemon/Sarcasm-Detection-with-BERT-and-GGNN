
# Sarcasm Detection with BERT and GGNN

 paper **Integrated transformer and gated graph neural networks detect sarcasm in user-generated content**  code
 [![DOI](https://zenodo.org/badge/854664840.svg)](https://doi.org/10.5281/zenodo.14755030)

## Overview

Sarcasm detection is an important task in sentiment analysis, especially in the context of social media, where sarcastic statements often contradict the literal sentiment of the text. This project tackles this challenge by integrating **BERT**, which captures deep contextual embeddings, with **GGNN**, which models complex dependencies through graph-structured data.

### Key Features:
- **BERT**: Extracts contextual embeddings of words, preserving semantic relationships.
- **GGNN**: Constructs a graph representation of the sentence to capture relations between entities.
- **Custom Layers**: Implements GCN, GAT, and GGNN layers to explore the effect of various graph-based learning techniques.
- **Ablation Studies**: Allows comparing different configurations (BERT-only, GGNN, etc.).
- **Multimodal Sentiment Analysis**: The project also includes extensions to multimodal datasets such as MOSI and IEMOCAP for multimodal sentiment detection.

## Project Structure

```
Sarcasm-Detection-with-BERT-and-GGNN-main/
│
├── README.md                    # Project documentation
├── data_utils.py                # Helper functions for data processing and dataset loading
├── train_ggnn.py                # Training script for the BERT+GGNN model
├── train_noly_BERT.py           # Training script for the BERT-only model
├── layers/                      # Custom layer implementations
│   ├── gcnlayer.py              # Graph Convolutional Network (GCN) layer
│   ├── gatlayer.py              # Graph Attention Network (GAT) layer
│   ├── ggnnlayer.py             # Gated Graph Neural Network (GGNN) layer
├── models/                      # Model definitions
│   ├── BERTOnly.py              # BERT-only model architecture
│   ├── bertgcn.py               # BERT + GCN hybrid model
│   ├── bertgat.py               # BERT + GAT hybrid model
│   ├── bertggnn.py              # BERT + GGNN hybrid model
│   ├── lstm_model.py            # LSTM-based sarcasm detection model
├── ablation experiment/         # Contains Jupyter Notebooks for ablation studies
│   ├── FNN.ipynb                # Ablation study: Feedforward Neural Network (FNN)
│   ├── GGNN.ipynb               # Ablation study: GGNN model
├── senticNet/                   # Resources for integrating SenticNet sentiment analysis
│   ├── senticnet_word.txt       # SenticNet word sentiment data
├── dataset/                     # Datasets for sarcasm detection
│   ├── headlines/               # Headline dataset for sarcasm detection
│       ├── train.txt            # Training data
│       ├── test.txt             # Test data
```

## Requirements

To set up the environment and run the experiments, you'll need the following:

- **Python** 3.7+
- **PyTorch** 1.7+
- **HuggingFace Transformers** for BERT-based models.
- **Graph Libraries**:
  - NetworkX for building and processing graphs.
  - Optional: PyTorch Geometric if working with large-scale graph neural networks.
- **Numpy**, **Pandas**, and **Matplotlib** for data manipulation and visualization.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```
- Download language model
```bash
python -m spacy download en
```
- Generate adjacency and affective dependency graphs
```bash
python graph.py
```

- Train the model. Optional arguments can be found in `train_ggnn.py`
```bash
python train_ggnn.py
```

## Theoretical Background

### BERT

**BERT (Bidirectional Encoder Representations from Transformers)** is a pre-trained language model that captures bidirectional context in a sentence. It transforms input sentences into contextual embeddings by understanding both left and right contexts simultaneously.

In this project, BERT is used to extract word-level representations, which are then fed into the GGNN to capture deeper, more complex relational features.

### GGNN

**GGNN (Gated Graph Neural Network)** is a variant of GNNs that uses gating mechanisms similar to LSTMs to propagate information across the graph. It is particularly useful for capturing relational information and dependencies in a sentence. By treating each word as a node and using dependency parsing to connect them, the GGNN processes graph-like structures in sentences to model the intricate dependencies in sarcastic text.

## Data Preparation

To use your own data for sarcasm detection, format it in `.txt` files where each line corresponds to an instance of the dataset, with the text and its corresponding label. The dataset should be placed in the `dataset/headlines/` directory.

Example format of a data file (`train.txt`):

```
"Text of the headline 1"    0
"Text of the headline 2"    1
...
```

Where `0` indicates non-sarcastic and `1` indicates sarcastic.

## Usage

### Training the BERT + GGNN Model

To train the hybrid BERT + GGNN model on your sarcasm dataset, use the following command:

```bash
python train_ggnn.py --train-file dataset/headlines/train.txt --test-file dataset/headlines/test.txt
```

This script will train the model using the specified training and test data, outputting the model's accuracy, F1-score, and loss metrics over time.

### Training BERT-Only Model

To train a BERT-only model for comparison, use the command below:

```bash
python train_noly_BERT.py --train-file dataset/headlines/train.txt --test-file dataset/headlines/test.txt
```

### Running Ablation Experiments

The ablation studies located in the `ablation experiment/` folder allow you to experiment with different models. You can launch the GGNN experiment via:

```bash
jupyter notebook ablation experiment/GGNN.ipynb
```

## Model Architectures

### BERT + GGNN Hybrid Model

The model uses BERT for word-level embeddings and passes these embeddings through a GGNN layer. The GGNN constructs a graph structure based on the dependencies in the sentence and refines the BERT embeddings to capture sarcasm-indicative patterns.

### BERT + GCN/GAT Models

In addition to GGNN, the models folder includes implementations for BERT + GCN and BERT + GAT models. These models can be used to compare different graph-based techniques for sarcasm detection.

## Evaluation and Results

After training, the model evaluates its performance on the test dataset, providing key metrics such as **accuracy**, **F1-score**, and **loss**. Results are displayed and stored in the form of plots that track the model's performance across epochs.

For detailed evaluation, you can load the results in any plotting library such as **Matplotlib** to visualize trends in accuracy and F1-score over time.

## Sentiment Integration with SenticNet

SenticNet is an external resource that can enhance sarcasm detection by providing sentiment context to each word in the sentence. The sentiment of words (positive, negative, neutral) is combined with the BERT embeddings to further boost performance. The data for this can be found in the `senticNet/` folder.

## Future Work

This project can be further improved by:
- **Multimodal Integration**: Extending the sarcasm detection to multimodal data (video, audio, text).
- **LSTM Models**: Replacing GGNN with LSTM to explore its effect on sarcasm detection.
- **Graph-Based Attention Mechanisms**: Implementing attention mechanisms at the graph level to enhance model performance.
- **Larger Datasets**: Experimenting on larger datasets to observe model scalability.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute as per the license terms.

 
