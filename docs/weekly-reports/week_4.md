# Time tracking

|date|time used|description|
|---|----|------------------|
|26.09|5h|full eigenfaces pipeline|
|total|5h||

# Report
This week I finished the main PCA flow: average face -> centering -> covariance matrix -> Jacobi eigendecomposition -> eigenfaces -> weights -> distances. I then ran the complete algorithm on the dataset. The first full run required 12 Jacobi sweeps and achieved approximately 92% classification accuracy. Next week I will implement tests for PCA and improve my documentation with demo examples of eigenfaces.