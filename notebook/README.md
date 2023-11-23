This folder has the code to execute DruID.
Note: Variant annotation is included as a pre-processing step and is done separately

Stage 1: Variant annotations
Annovar annotations can be generated using the files data\_cleaning\_*.ipynb files. These are further combined and processed using the clinvar\_annovar\_gpd\_annotated\_data\_preprocessing.ipynb file.

Stage 2: Unsupervised Domain Invariant Representation Learning
Pretraining the VAEs with CORAL loss is done through DruID\_pretraining.ipynb

Stage 3: Multi task learning for drug response prediction
The pretrained VAEs are attached to an MTL network and trained using DruID\_training.ipynb. 

Inference:
From a trained DruID model, the inference can be done using inference.ipynb