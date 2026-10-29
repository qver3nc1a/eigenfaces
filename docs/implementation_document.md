Dataset: Olivetti Faces

Number of persons: 40

Training images: 280 (7 per person)

Validation images: 120 (3 per person)

Image size: 64x64

Split seed: 42

Eigenface selection: fixed k or variance percentage (default 95%)

## Different numbers of eigenfaces (k)
| k |accuracy|correct|total|jacobi sweeps|
|---|--------|-------|-----|-------------|
| 5 |73.33%|88|120|12
| 10|85.83%|103|120|12
| 20|90.00%|108|120|12
| 40|91.67%|110|120|12
| 80|91.67%|110|120|12
| 95%|91.67%|110|120|12|