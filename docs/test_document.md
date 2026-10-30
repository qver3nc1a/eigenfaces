# Test document
The project is tested using unit tests implemented with pytest.

## Jacobi tests
The Jacobi eigenvalue algorithm is tested using 20 symmetric matrices of size 20x20. The matrices are generated randomly, using 4 different size scales (1e-6, 1, 1e6, 1e8) and 5 different seeds (40, 41, 42, 43, 44). To make sure the generated matrices are symmetric, the following formula is used: A = scale * (B + np.transpose(B)), where B is a random matrix of size 20x20.

Each test runs the Jacobi algorithm with tolerance of 10^-16 and a maximum of 50 sweeps. The tests verify that the algorithm runs in less than 15 sweeps and that the calculated eigenvalues and eigenvectors are correct. The correctness of the eigenvalues and eigenvectors is checked with the help of the eigenvalue equation: $A*v_i=\lambda_i*v_i$. The equation is checked with numpy.allclose() to allow for small floating-point errors.

## PCA tests
The PCA tests use small numpy arrays with known values, allowing the functions to be checked without loading the Olivetti dataset used in this project.

### Average face (test_average_face())
The average face test uses a small 3x3 sample matrix and verifies that the average face is calculated and the faces centered correctly.

### Covariance matrix (test_covariance_symmetric())
The covariance matrix test calculates the covariance matrix for a 3x3 centered matrix and verifies that the covariance matrix is symmetric and of the right dimensions.

### Nearest neighbor (test_nearest_neighbor())
The nearest neighbor test verifies that the nearest-neighbor classifier works correctly on small sample categories. The training data used is [0.0, 0.0], [20.0, 0.0] with respective labels A and B. The validation data is [0.1, 0.3], [15.0, 22.0]. The expected predictions are [A, B] and the test verifies that both categories are assigned correctly.

### PCA face recognition pipeline (test_pca())
The test verifies that the main recognition steps work together well. The training data consists of 6 two-dimensional samples divided into two classes, A and B. Two samples are used for validation.
The test performs the folowing operations:

- Calculates the mean vector and centers the training data
- Calculates one eigenfaces using the eigenfaces() function
- Projects the training and validation samples onto the eigenface space
- Classifies the validation samples
- Compares predicted labels with expected labels

The expected labels are A and B. The test validates whether the functions produce the correct prediction when working together on a small sample dataset.

## Running the tests
The tests use synthetic data and therefore it is not required to load the Olivetti dataset prior to running them.
The tests can be executed from the project root using Poetry:
```bash
poetry run pytest -v
```

Only Jacobi tests:
```bash
poetry run pytest tests/test_jacobi.py -v
```

Only PCA tests:
```bash
poetry run pytest tests/test_pca.py -v
```

## Results
- Number of tests: 35
- Passed: 35

### Coverage report
```bash
Name            Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------
src/jacobi.py      59      1     14      1    97%   84
src/pca.py         26      0      2      0   100%
-----------------------------------------------------------
TOTAL              85      1     16      1    98%
```

The coverage report can be generated with:
```bash
poetry run coverage run --branch -m pytest
```
And displayed with:
```bash
coverage report -m
```

## Olivetti evaluation
In addition, the program is evaluated on the original faces from the Olivetti Faces dataset.

### Dataset properties & Evaluation
Number of persons: 40

Training images: 280 (7 per person)

Validation images: 120 (3 per person)

Image size: 64x64

Split seed: 42

Eigenface selection: fixed k or variance percentage (default 95%)

| k |accuracy|correct|total|jacobi sweeps|
|---|--------|-------|-----|-------------|
| 5 |73.33%|88|120|12
| 10|85.83%|103|120|12
| 20|90.00%|108|120|12
| 40|91.67%|110|120|12
| 80|91.67%|110|120|12
| 95%|91.67%|110|120|12|