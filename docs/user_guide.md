# User Guide

## Installation
Clone the repository:

```bash
https://github.com/qver3nc1a/eigenfaces.git
cd eigenfaces
```

Install the dependencies with Poetry:
```bash
poetry install
```

## Data
The program uses the [Olivetti Faces](https://scikit-learn.org/0.19/datasets/olivetti_faces.html) dataset. Run the [loader file](../data/load_olivetti.py) from the project root with the following command:
```bash
python data/load_olivetti.py
```

## Running the Program
Run the program from the project root:
```bash
python src/pca.py
```