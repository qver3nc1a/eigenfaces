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
- Number of tests: 24
- Passed: 24

### Coverage report
|Name            | Stmts | Miss | Cover |
|----------------|-------|------|-------|
|src/dataset.py  |   33  |  29  |  12%  |
|src/jacobi.py   |   44  |   2  |  95%  |
|src/pca.py      |   48  |  21  |  56%  |
|TOTAL           |  125  |  52  |  58%  |

The coverage report can be generated from the project root:
```bash
poetry run pytest --cov=src tests/
```