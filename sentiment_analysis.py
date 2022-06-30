import matplotlib.pyplot as plt    
from nltk.tokenize import wordpunct_tokenize
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns



class sentiment_analysis:

  def __repr__(self):
    return "Sentiment Analyst"
    

  def GraphCommonBar(self, df, review_type, stop_words, common_axs, position):

    #The next 6 lines of code are from https://www.kaggle.com/edhirif/word-cloud-alternative-using-nltk
    review_str = df["Review"].str.cat(sep = ' ') #Takes each row of the reviews dataframe and puts it into a single string. Each review is separated with a space.
    list_of_words = [i.lower() for i in wordpunct_tokenize(review_str) if i.lower() not in stop_words and i.isalpha()] #Lowercases each word in every review if it is not in the list of stop words and all the characters in the word are letters.
    wordfreqdist = nltk.FreqDist(list_of_words)#Counts how frequent each word is.
    mostcommon = wordfreqdist.most_common(25) #creates a list of the 25 most common words.
    common_axs[position].barh(range(len(mostcommon)),[val[1] for val in mostcommon])
    common_axs[position].set_title(review_type + ' Reviews')
    plt.sca(common_axs[position])
    plt.yticks(range(len(mostcommon)),[val[0] for val in mostcommon],fontsize=15)

  def makePredictions(self,X_train, y_train, X_test, y_test):
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    predictions = lr.predict(X_test)
    self.ConfusionMatrix(y_test,predictions)
  
  def ConfusionMatrix(self,y_test,predictions):
    cm=confusion_matrix(y_test,predictions)
    #The code used to create the confusion matrix is from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    fig, ax= plt.subplots(figsize=(10,7.5))
    sns.set(font_scale=1.5) # Adjust to fit
    sns.heatmap(cm, cmap="coolwarm",annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted Ssentiment', fontsize=15);ax.set_ylabel('Actual Sentiment',fontsize=15); 
    ax.set_title('Confusion Matrix',fontsize=15); 
    ax.xaxis.set_ticklabels(['Positive', 'Negative'],fontsize=15); ax.yaxis.set_ticklabels(['Positive', 'Negative'],fontsize=15);
    plt.show()
    
    self.confusion_matrix_metrics = classification_report(y_test,predictions)