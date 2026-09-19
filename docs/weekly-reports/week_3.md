# Time tracking

|date|time used|description|
|---|----|------------------|
|16.09|1h|finish jacobi, read pytest docs and write first tests|
|18.09|1,5h|fix jacobi sweeps, write some jacobi tests|
|19.09|10h|fix jacobi, text jacobi, find and load dataset, start pca|
|total|12,5h||

# Report
This week I finished the Jacobi algorithm. Later I found some issues in my implementation, for example, my algorithm had a big issue where it would sometimes take the maximum 50 sweeps to complete. The problem was that in cases where the diagonal values were much greater than off-diagonal values, my old computation completely disregarded the change to off-diagonal non-zero values, which led to the squared off-diagonal sum never falling below the tolerance. I also empirically tested my solution and found that a tolerance of 1e-16 was fitting due to the very tight precision requirement of np.allclose(), which I used in my tests to verify that the calculated eigenvalues and eigenvectors are correct. I then wrote a very comprehensive test for my Jacobi implementation. The test generates 5 20x20 matrices with values of 4 different scales. Then it verifies that no matrix requires more than 15 sweeps and that all calculated eigenvalues and eigenvectors are correct.

I still believe that my implementation of Jacobi can be improved, but for now i deemed it precise enough and moved on with my project. I found the Olivetti faces dataset, loaded it, parsed the images and split them into train and validation sets. There is still an issue with dataset.py that I have not yet had time to solve. My guess is that something is wrong with the way I split data.

While my data was loading I had a little time to get started on the PCA. I calculated the average face and substracted it from all images.

As soon as I am done fixing whatever is wrong with my dataset.py, I will move on to create the covariance matrix (C=AA^T) and make use of my Jacobi algorithm to calculate the eigenvalues and eigenvectors for it. Then I will calculate the weights and distances.