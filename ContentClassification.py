import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense
import keras.backend as K


def bow_fea():

    import nltk

    f=open('mmr3.txt','r')

    content=f.read().splitlines()

    #print content

    fw=open('feature.txt','w')

    bow = []
    from nltk.stem import WordNetLemmatizer

    for sentence in content:

        first=sentence.split('\t', 1)[1]

        sentence=sentence.split('\t', 1)[0]
        

        nouns = [] 

        nouns.append(first)   

        words = nltk.word_tokenize(str(sentence))

        for word,pos in nltk.pos_tag(words):

            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):

                word = word.replace('\x92',"")
                word = word.replace('\x93',"")
                word = word.replace('\x94',"")
                word = word.replace('\x96',"")
                word = WordNetLemmatizer().lemmatize(word)
                nouns.append(word)

                bow.append(word)

        

        #print nouns

        fea=' '.join(nouns)

         

        fw.write(fea)

        fw.write('\n')

        bow = list(set(bow)) 

    return bow



    fw.close()

    f.close()

def keras_fea():

    import nltk
    from nltk.stem import WordNetLemmatizer

    #wnl = WordNetLemmatizer()
    bow = bow_fea()

    fr=open('feature.txt','r')

    content=fr.read().splitlines()
    xtrain=[]
    ytrain=[]

    for sentence in content:

        fea_vector=[]

        for i in bow:

            fea_vector.append(0)

        label=sentence.split(' ', 1)[0]

        fea=sentence.split(' ', 1)[1]

        token = nltk.word_tokenize(fea)

        for word in token:

            pos=bow.index(word)

            fea_vector[pos]=1
      
        #fea_vector.append(label)
        if(label == "relevant"):
            ytrain.append(1)
        else:
            ytrain.append(0)
        #print fea_vector

        xtrain.append(fea_vector)
    

    return xtrain,ytrain

    fr.close()
    
def bow_fea_test():

    import nltk

   # f=open('sample.txt','r')

    f=open('mmr_test.txt','r')

    content=f.read().splitlines()

    #print content

    fw=open('feature_test.txt','w')

    bow_test = []
    from nltk.stem import WordNetLemmatizer

    for sentence in content:

        first=sentence.split('\t', 1)[1]

        sentence=sentence.split('\t', 1)[0]
        

        nouns = [] 

        nouns.append(first)   

        words = nltk.word_tokenize(str(sentence))

        for word,pos in nltk.pos_tag(words):

            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):

                word = word.replace('\x92',"")
                word = word.replace('\x93',"")
                word = word.replace('\x94',"")
                word = word.replace('\x96',"")
                word = WordNetLemmatizer().lemmatize(word)
                nouns.append(word)

                bow_test.append(word)

        

        #print nouns

        fea=' '.join(nouns)

         

        fw.write(fea)

        fw.write('\n')

   

    bow_test = list(set(bow_test)) 

    #print ''          #print featureset

    #print ('Bag of Words are')

    #print bow

    return bow_test



    fw.close()

    f.close()

def keras_fea_test():

    import nltk
    from nltk.stem import WordNetLemmatizer

    bow_test = bow_fea_test()
   
    fr=open('feature_test.txt','r')

    content=fr.read().splitlines()
    xtest=[]
    ytest=[]

    for sentence in content:

        fea_vector=[]

        for i in bow_test:

            fea_vector.append(0)

        label=sentence.split(' ', 1)[0]

        fea=sentence.split(' ', 1)[1]

        token = nltk.word_tokenize(fea)

        for word in token:

            pos=bow_test.index(word)

            fea_vector[pos]=1
      
        #fea_vector.append(label)
        if(label == "relevant"):
            ytest.append(1)
        else:
            ytest.append(0)
        #print fea_vector

        xtest.append(fea_vector)
    

    return bow_test,xtest,ytest

    fr.close()


def simple_nn():
    bow_test,x_test,y_test = keras_fea_test()
    model = Sequential()
    keras.initializers.Zeros()
    # 2 layers
    model.add(Dense(64, input_dim=len(bow_test), activation='relu')) # 64 no.of .neurons
    model.add(Dense(32, input_dim=len(bow_test), activation='relu'))
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', f1_score, precision, recall ])
    
    model.fit(x_test, y_test, nb_epoch=10000, verbose=2)
    print model.predict(x_test)
    return model


def performance():
   
    X_test=[[0,1],[1,1]]
    y_test=[[1],[1]]
    
    model=simple_nn()
    
    y_pred=model.predict(X_test)
    print y_pred
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test Accuracy:', score[1])
    print('Test F1-measure:', score[2])
    print('Test Precision:', score[3])
    print('Test Recall:', score[4])
    

def metric(y_test, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_test, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    return c1,c2,c3
    
def precision(y_test, y_pred):
    m=metric(y_test,y_pred)
    # How many selected items are relevant?
    precision = m[0] / m[1]
    return precision

def recall(y_test, y_pred):
    m=metric(y_test,y_pred)
    # How many relevant items are selected?
    recall = m[0] / m[2]
    return recall

def f1_score(y_test, y_pred):
    p=precision(y_test, y_pred)
    r=recall(y_test, y_pred)
    # Calculate f1_score
    f1_score = 2 * (p * r) / (p + r)
    return f1_score