# Time tracking

|date|time used|description|
|---|----|------------------|
|10.10|6h|pca tests, test document|
|total|6h||

# Report
This week I wrote tests for the PCA pipeline and worked on the test document. Both the tests and the document are still in progress. One of the tests (test_project) passes but it should fail, so I will investigate and fix this next week.
I also noticed that the Jacobi rotations are not efficient as the entire matrix is recalculated with each rotation, even though only a small portion of the values is affected. I will look into improving the efficiency of this implementation.

Next week I will test the algorithm on actual data and compare my results to those in the original paper. 