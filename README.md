
[![Watch Video](https://github.com/aman010/A4-Bert/blob/main/Screenshot%20from%202025-02-23%2015-57-42.png)](https://youtu.be/xoO7oBN6fGM)


The above videos links is about how the app works, in this work we are not able to deploy the web app, possibly tried to push the model to huggyface and run it with streamlet 
which is still not possoble because of size limitation. 


![BERT Model Screenshot](https://raw.githubusercontent.com/aman010/A4-Bert/main/Screenshot%20from%202025-02-23%2012-30-56.png)


With the above loss of pretrain Bert model we observe that it over optimzed very quickly with the training , we took 100000 samples from openweb text.
Tried few experiments
  *  number of layer to 8
  *  added dropout to 0.3
  *  added ReduceLROnPlateau
  *  added weighted decay (L2)
  *  This is the best we can get the above one is number of layers 8
  *  The sample is non random of fisrt 100000 samples the training took almost an 30-45 mins
  *  If enough time is left we will try to do other experiements to optimize the pretrain model
  *  with this lets move to siamese network







| Model                  | Accuracy_MNLI | Traning Loss MNIL | Traning Time | sample size |    
|------------------------|---------------|-------------------|--------------|-------------|
| Bert Pretrain siamese  | 0.997         |   2.71            |     <40min   |   > 1000    |
