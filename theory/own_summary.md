
1. Prepare dataset

 - import dataset (ex. MNIST dataset)
 - split the data into the train and test datasets
 - preprocess the data
   - reshape the data by doing like resizing images or chainging images to B&W

2. Build the model

 - set up the layers
   - flatten the data or do embeddings
   - dense
   - drop out
 - compile the model with the following settings
   - Optimizer —This is how the model is updated based on the data it sees and its loss function.
   - Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
   - Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

3. Train the model

    - feed the training data to the model
    - evaluate the accuracy of the model with test (validation) data
      - check if the model is overfitted

4. 