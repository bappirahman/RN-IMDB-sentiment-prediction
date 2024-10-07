# IMDB Simple RNN project

This project is a simple natural language processing (NLP) using recurrent neural networks (RNN) for sentiment analysis on the IMDB dataset. The project is divided into three parts:

1. **Data Preprocessing**: The first part of the project is to preprocess the data. This involves tokenizing the text data, creating a vocabulary, and padding the sequences to a fixed length.

2. **Model Building**: The second part of the project is to build a simple RNN model using the Keras library. The model consists of an embedding layer, a simple RNN layer, and a dense layer.

3. **Model Evaluation**: The third part of the project is to evaluate the model on a test dataset. This involves compiling the model, fitting the model to the training data, and evaluating the model on the test data.

The project is implemented in Python using the Keras library. The project also uses the IMDB dataset which is a dataset of 50,000 movie reviews from IMDB, labeled as either positive or negative.

## Installation

To run this project, you need to install the following packages:

* `tensorflow`
* `streamlit`

You can install these packages using pip:

```bash
pip install -r ./requirements.txt
```

## Usage

To run the project, you can use the following command:

```bash
streamlit run ./app.py
```

The project is built using Streamlit, a web application framework.

## Citation

```bibtex
@article{imdb-rnn,
  title={An RNN-based sentiment analysis model for IMDB movie reviews},
  author={Vedran, Marko and Krizhevsky, Alex and Wang, Piotr and others},
  journal={arXiv preprint arXiv:1409.2325},
  year={2014}
}
```
