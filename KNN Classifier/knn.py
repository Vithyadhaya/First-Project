from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
import re
import numpy
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import csv

stop_words = []
samp_list = []
#Preparing Test Data
test_corp_root = 'Test_set' 
testcorpus = PlaintextCorpusReader(test_corp_root, '.*')
porter_stemmer = PorterStemmer()
for x in range(0, len(testcorpus.fileids())):    
    stop_words = stopwords.words("english")    
    corp_data = [re.sub('[^A-Za-z0-9.]+', ' ',word).strip() for word in testcorpus.words(testcorpus.fileids()[x]) if word not in stop_words]    
    corp_data = ' '.join(str(e) for e in corp_data)    
    samp_list.append(corp_data)

categ_dict = {'test_1.txt':'Regulatory Update','test_2.txt':'Press Release','test_3.txt':'Regulatory Update','test_4.txt':'Regulatory Update',
              'test_5.txt':'Stock Update','test_6.txt':'Press Release','test_7.txt':'Market Opinion'}
art_i = []
class_i = []
#Conversion of Train Data into Single Input File
corpus_root = 'Train_set'

newcorpus = CategorizedPlaintextCorpusReader(corpus_root, r'.*\.txt', cat_pattern=r'(\w+)/*')

myfile = open('Input_Article_Data.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,lineterminator="\n")

for category in newcorpus.categories():
    for fileid in newcorpus.fileids(category):
        #print fileid,category	
        data1 = (newcorpus.raw(fileid).encode('utf-8')).replace(","," ")
        data_list = [data1,category]
        wr.writerow(data_list)
        
myfile.close()


#Reading of Train Data as Lists
with open('Input_Article_Data.csv', 'r') as f:    
    for line in f.readlines():
        l,name = line.strip().split(',')
        l=(re.sub('[^A-Za-z0-9.]+', ' ',l)).lower()
       # l=porter_stemmer.stem(l) #Reduces Accuracy From 50% To 37%
        if(name != "Category"):
            art_i.append([l])            
            class_i.append(name)
f.close()

data = art_i

labels = class_i

#vec = CountVectorizer()  # count word occurrences
vec = TfidfVectorizer()

X = vec.fit_transform([' '.join(row) for row in data])
#clf = MultinomialNB()  # very simple model for word counts
clf = neighbors.KNeighborsClassifier(n_neighbors=2)
clf.fit(X, labels)# Training the KNN Classifier with Train Data Set
count = 0
for x in range(0, len(testcorpus.fileids())):
    new_data = [samp_list[x]]
    new_X = vec.transform([' '.join(new_data)])
    predict_label = clf.predict(new_X)[0]
    predict_label = (re.sub('[^A-Za-z0-9.]+', ' ',predict_label)).strip()
    print 'File Name:%s Predicted: %s   Actual:%s'%(testcorpus.fileids()[x],predict_label,categ_dict[testcorpus.fileids()[x]])
    #print 'File Name:%s Predicted: %s  '%(testcorpus.fileids()[x],predict_label)
    if predict_label == categ_dict[testcorpus.fileids()[x]]:
        count = count+1
total = len(testcorpus.fileids())
print 'Accuracy (Percentage): %f '%((count/total)*100)

