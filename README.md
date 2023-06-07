# Music Mood Ring

## Intro
With the popularization of digital music streaming services, music characterization has grown in popularity so that users can simply request a mood or genre and receive song recommendations. Our project tackles a multi-class classification problem and a regression task; Using supervised learning to predict the mood of a song based on its lyrics, we have created a model to help users find music based on a desired mood category and a musical positivity score (valence). The model references the paper Music Mood Detection Based On Audio And Lyrics With Deep Neural Net, which we have modified to only consider lyrics. This project could be used to make playlists on Spotify (or other music platforms) or radio genres. Ultimately, it would help users find music in a mood category of their choosing.

## Methodology
### Model

We used the Tensorflow framework to implement a model architecture consisting of a word embedding layer, 1D convolution, max pooling, an LSTM/GRU (tested both), a dense layer with relu/tanh activation (depending on the task), dropouts of 0.5, and a dense output layer with softmax/sigmoid activation (depending on the task). Our models were fairly similar across both our regression and classification models however we did change activation functions and layer sizes accordingly. (See Figure 1) This diagram displays a basic representation of our models. Each model however has a few unique features as discussed.

### Data

The original paper used both audio and lyrics. To simplify this problem, we only used lyrics. Rather than using the Million Song Dataset as suggested by the paper, we used two different datasets; one for each task. For the classification task, we used a dataset that consisted of 1,160 songs and three different categories for moods: sadness, tension, and tenderness. For our regression taks, we used a dataset with over 150,000 lyrics labeled with valence (positivity score of a song on a scale of 0 to 1) values gathered using Spotify API.

### Preprocessing

For our preprocessing, we had two separate datasets to consider. For both datasets, we simplified the dataset to only consider the lyrics and the label (an emotion label for classification and a valence score for our regression task). We used the nltk word corpus to help clean unnecessary data in both sets. For our classification dataset, we trimmed each song down to the first 50 lyrics, balanced out the dataset so that each emotion had an even number of songs, and one hot encoded the labels. Additionally, we removed outliers in song length by finding the mean and standard deviation. We also added padding so that every song was the same length and we could eventually convert our data into tensors. We split the dataset into training and testing sets (using an 80%/20% split). We used a very similar methodology for our regression dataset, but did not one hot encode our labels and simply left them as valence scores.

## Hyperparameters/Optimization
### Training + Testing

For both tasks, we used Stochastic Gradient Descent (SGD) as our optimizer. We started with Adam as our optimizer, however, when we encountered some overfitting, we switched to SGD. At first, we used categorical cross entropy as our loss for the classification task. We also played around with making our own loss function that would penalize certain incorrect classes more harshly when we found our model guessing the same class every time. For the regression taks, we used mean squared loss. The paper used R2 scores based on valence and arousal to assess the performance of the regression model. For the regression task, we used R2 scores based on valence only. For the classification task we used an accuracy based on the percentage of correct classifications. Both models were trained on varying numbers of epochs with various hyperparameters to find the best overall results (final values are summarized above).

## Results
### Classification

The highest accuracy we achieved was .536 for 100 epochs as we struggled to move it past this level. Upon inspection, we discovered that the model guessed the same class for every song and correctly predicted moods 50% of the time. We attempted to fix this issue, but ultimately came to the conclusion that the dataset was not large enough to generate a better result (discussed further below). 

### Regression

After adjusting optimizers and hyperparameters and training for 50 epochs, our data received a final R2 score of .2067 and a loss of .037. Our testing data received an average R2 score of .137. For reference, the R2 scores in the original paper averaged .177, with their best results reaching a score of .5 for lyrics. We replicated the paper as closely as possible but were not able to quite reach the desired results.

### Future Work

For classification, we would like to test this model on a larger dataset to see if it will predict more accurately, however, we were unable to find an adequate dataset to attempt this while working on the project. Future work might utilize a dataset that has more diverse emotions and lyrics to mitigate our issue with similarity of emotions. For regression, we would like to use a pre-trained word embedding, such as word2vec or BERT. We tried to implement this towards the end of our project, but were unsuccessful given the amount of time we had. Finally, we could train with an even larger dataset or using more than one label, such as adding arousal. Additionally, in the future we would want to add audio data to our set in the hopes of increasing prediction accuracy. This would be a form of data we have not worked with yet and would prove to be a fun challenge.

## More 
See our paper for more information and detailed results! 
It is here: https://docs.google.com/document/d/1d1i7YizwJB2cLHz_VM3CIthxRhxcG_OisTWcJ18-It0/edit?usp=sharing

Please note: Our larger dataset is too large to upload to git. Please download it yourself and you will be able to run our code. Here is the link:
 https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence?resource=download.