# ACDGNN
Source code for ["Attention-based Cross Domain Graph Neural Network for Prediction of Drug-Drug Interactions"](https://academic.oup.com/bib/article-abstract/24/4/bbad155/7167644)
   
 

## Required packages
* Python == 3.6
* [Tensorflow](https://www.tensorflow.org/) == 1.14



## File description


-  data —— contains the data used in our paper.

-  myutils —— contains some utility functions.

- DSRADDI.py —— main part of codes for model training and testing.






## Usage example 
    python DSRADDI.py --epoch 10 --batch_size 2048 --ad 0.2 -s 'no_inductive/data_0'

Arguments:

    --epoch Number of training epoches.
    --batch_size Batch size of DDI triplets.
    --ad dropout rate.
 	-s Choose one split dataset to train model.
