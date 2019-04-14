Workshop
========================================================
author: Vlad
date: 19/05/2001
font-family: 'Helvetica'
css: presstyle.css


Overview
========================================================

1. Computational propaganda (20 minutes)
  - social bots, trolls, cyborgs
  - methods of identification
  - list of Reddit trolls
  - explore Reddit dataset
2. Text as data (40 minutes)
  - Document-term matrix
  - Text processing
  - Example with Reddit submission titles
  
***
3. Basics of Machine Learning (1 hour)
  - Lasso regression
  - Random forests
  - Building an Ensemble learner
  - Identifying trolls on Reddit
4. Pipeline to apply to your own data (30 minutes+)


Computational propaganda - TBD
========================================================

### Bullet 1
- Bullet 2
- Bullet 3


Text as Data
========================================================
<hr>
<p></p>
- Corpus = a collection of texts/documents
- Corpora often come with metadata
- Bag of words approach
  - Count of words for each document in corpus
  - Discards word order, ignores word meaning
  - Purpose: reduction of complexity


Computational text analysis
========================================================
![alt text](pics/img01.png)
<br><span clas="at75"><i>(Grimmer and Stewart 2013:2)</i></span>

Not covered in this seminar
========================================================
<hr><p></p>
- Part of Speech (POS) tagging
- Named entity recognition (NER)
- Word Sense Disambiguation (WordNet)
- Word embeddings (word2vec)





Tokenization
========================================================
<hr>
Text: <span class = "sky">"X and Y are 2 Kremlin trolls! Trolling day and night for a few rubles."</span>
- Tokens
  - Sentences <span class = "sky">["X and Y are 2 Kremlin trolls !","Trolling day and night for a few rubles."]</span>
  - Words / Unigrams <span class = "sky">["X", "and", "Y", "are", "2", "Kremlin", "trolls", "!","Trolling", ..., "rubles","."]</span>
  - Letters
- N-grams
 - Bigrams <span class = "sky">[...,"Kremlin trolls","a few","few rubles",...]</span>
 - Trigrams <span class = "sky">[...,"day and night",...]</span>
 - Skip-grams <span class = "sky">["are Kremlin trolls",..."trolling for rubles",...]</span>



Text pre-processing
========================================================
class: small-code
<hr>

Text: <span class = "sky">"X and Y are 2 Kremlin trolls! Trolling day and night for a few rubles."</span>
- remove capitalization <br><span class="sky">Putin -> putin</span>
- remove stopwords <br><span class="sky">!["and","a","for","few","here"]</span>
- (remove non-words: numbers, punctuation) <span class="sky"><br>["2","!","."]</span>
- combine similar terms (stemming, lemmatization)
- create a Document-Term Matrix


Stemming vs Lemmatization
========================================================
<hr>
Many-to-one mapping from words to stem/lemma<br>

### Stemming 
- reducing inflected words to their word stem
- cutting off common suffixes 
<br><span class="sky">trolling -> troll<br>trolls -> troll<br>rubles -> rubl</span>

### Lemmatization
- based on morphological analysis of each word
<br><span class="sky">are -> be</span>


Document-Term Matrix (DTM)
========================================================
<hr><span class = "at75">
- Each document = a vector of word counts<br>
- N (number of documents) rows X P (number of features) columns
- Discards word order, ignores word meaning
- Large and sparse matrix (many 0s)
- Main Input for Supervised Machine Learning 

<span class = "sky">
Text1: "X and Y are 2 Kremlin trolls! Trolling day and night for a few rubles."<br>
Text2: "Kremlin's trolls are here tonight"<br>
Text3: "Days are few and long"</span>
<table class="tg">
  <tr>
    <th></th>
    <th>x</th>
    <th>and</th>
    <th>y</th>
    <th class ='tg-emp'>are</th>
    <th class ='tg-emp'>kremlin</th>
    <th class ='tg-emp'>troll</th>
    <th>day</th>
    <th>night</th>
    <th>for</th>
    <th>a</th>
    <th>few</th>
    <th>rubl</th>
    <th>here</th>
    <th>tonight</th>
    <th>long</th>
  </tr>
  <tr>
    <td>text1</td>
    <td>1</td>
    <td>2</td>
    <td>1</td>
    <td class ='tg-emp'>1</td>
    <td class ='tg-emp'>1</td>
    <td class ='tg-emp'>1</td>
    <td>2</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>text2</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td class ='tg-emp'>1</td>
    <td class ='tg-emp'>1</td>
    <td class ='tg-emp'>1</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>1</td>
    <td>1</td>
    <td>0</td>
  </tr>
    <tr>
    <td>text3</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>1</td>
  </tr>
</table>


Normalization and Weighting DTM
========================================================
left: 50%
<hr>
### Normalization (relative frequencies)
- proportions of the feature counts within each document
$$\mu_{j}  = \frac{\sum_{i=1}^{P}tf_{ij}}{N_{j}}$$

### TF-IDF weighting
- Term Frequency - Inverse Document Frequency
$$w_{i,j} = tf_{i,j}\cdot log(\frac{N}{df_{i}})$$

<table class="tg2">
  <tr>
    <th></th>
    <th>x</th>
    <th>y</th>
    <th>kremlin</th>
    <th>troll</th>
    <th>day</th>
    <th>night</th>
    <th>rubl</th>
    <th>tonight</th>
    <th>long</th>
  </tr>
  <tr>
    <th>text1 - normalized</th>
    <th>0.125</th>
    <th>0.125</th>
    <th>0.125</th>
    <th>0.250</th>
    <th>0.125</th>
    <th>0.125</th>
    <th>0.125</th>
    <th>0</th>
    <th>0</th>
  </tr>
  <tr>
    <th>text1 - TF-IDF</th>
    <th>0.477</th>
    <th>0.477</th>
    <th>0.176</th>
    <th>0.352</th>
    <th>0.176</th>
    <th>0.477</th>
    <th>0.477</th>
    <th>0</th>
    <th>0</th>
  </tr>
  <tr>
    <th>text1 - TF-IDF normalized</th>
    <th>0.059</th>
    <th>0.059</th>
    <th>0.022</th>
    <th>0.044</th>
    <th>0.022</th>
    <th>0.059</th>
    <th>0.059</th>
    <th>0</th>
    <th>0</th>
  </tr>
</table>






Machine learning
========================================================
class: at75 
<hr>
#### Goal: building a statistical model for predicting or estimating an output based on one or more inputs
  - <b>X</b> = Inputs / Features
    - ~ Independent variables
  - <b>y</b> = Output / Response
    - ~ Dependent variable<br>
    
<table>
  <tr>
    <th></th>
    <th colspan="3">Inputs (<b>X</b>)</th>
    <th>Output (<b>y</b>)</th>
    <th></th>
  </tr>
  <tr>
    <td>x1</td>
    <td>1</td>
    <td>0</td>
    <td>2</td>
    <td>TROLL<br></td>
    <td>y1</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>0</td>
    <td>0</td>
    <td>5</td>
    <td>not troll</td>
    <td>y2</td>
  </tr>
  <tr>
    <td>x3</td>
    <td>0</td>
    <td>1</td>
    <td>-2</td>
    <td>not troll</td>
    <td>y3</td>
  </tr>
</table>

***
<br>
### Supervised machine learning
- learn mapping from inputs to outputs given a labeled set of input-output pairs
- use mapping on new (unlabeled) inputs
- task depends on measurement level of Y
  - Y is categorical: classification 
  - Y is continuous: regression
<br>

### <br>Unupervised machine learning
- find patterns in data given only inputs
- e.g. clustering

Supervised machine learning for document classification
========================================================
<hr>
- documents are automatically coded according to previously defined content categories by training a computer to replicate the coding decisions of humans
- hand coding is used to train, or supervise, statistical models to classify texts in pre-determined categories.
- goal is to classify unknown documents by using the hand-coding to “train” or “supervise” statistical models to classify the remaining documents
- can use the document-term matrix as input, along with available metadata


Training and test set
========================================================
class: at75
<hr>
- Split the set of observations into <b> a training set</b> and <b> a test set</b>
- Learn the relationship between input and output using <b>only</b> training data
- Check the prediction error on test data, see if overfitting:
![alt text](pics/training_test_error.png)



Cross-validation
========================================================
class: at75
- randomly divide the set of observations into <i>k</i> groups (5,10) of equal size
- used for model selection
  - for k = 1..K, train on all folds but kth, validate on kth
  - average of k estimates of test error
<span class = "at150">
$$CV_{k}=\frac{1}{k}\sum_{i=1}^{k}{Err_{i}}$$ $$Err_{i}=I(y_{i} \neq \hat{y}_{i})$$
</span>
<table class = "tg3">
  <tr>
    <th></th>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
    <th>5</th>
  </tr>
  <tr>
    <td>k=1</td>
    <td><b><b>VALIDATE</b></td>
    <td>train</td>
    <td>train</td>
    <td>train</td>
    <td>train</td>
  </tr>
  <tr>
    <td>k=2</td>
    <td>train</td>
    <td><b>VALIDATE</b></td>
    <td>train</td>
    <td>train</td>
    <td>train</td>
  </tr>
  <tr>
    <td>k=3</td>
    <td>train</td>
    <td>train</td>
    <td><b>VALIDATE</b></td>
    <td>train</td>
    <td>train</td>
  </tr>
  <tr>
    <td>k=4</td>
    <td>train</td>
    <td>train</td>
    <td>train</td>
    <td><b>VALIDATE</b></td>
    <td>train</td>
  </tr>
  <tr>
    <td>k=5</td>
    <td>train</td>
    <td>train</td>
    <td>train</td>
    <td>train</td>
    <td><b>VALIDATE</b></td>
  </tr>
</table>
- can also do repeated cross-validation

Model evaluation
========================================================
class: at75
<hr>

Many different methods/classifiers
- K nearest neighbours
- Naive Bayes
- Support vector machines
- Regularized logistic regression
  - Ridge Regression
  - <b> The Lasso </b>
  - Elastic net
  
*** 
<br><br>
- Generalized additive models (GAMs)
- Tree-based methods
  - Classification trees (CART)
  - Bagging
  - Boosting
  - <b> Random forests </b>

*** 
There is no single best model that works optimally for all datasets


Model evaluation / Metrics 
========================================================
<hr>
need to finish
![accuracy](pics/perf_acc.png) 
![precision](pics/perf_precision.png) 
![recall](pics/perf_recall.png) 

Lasso regression
========================================================
TBD

Random forests
========================================================
TBD

Ensemble classifier
========================================================
TBD


