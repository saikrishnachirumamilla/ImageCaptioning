from google.colab import drive
drive.mount('/content/gdrive')

from google.colab import drive

drive.mount('/content/gdrive')

root_path = 'gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data'

def fileOpen(filepath):
  fileObject=open(filepath,'r')
  fileContent=fileObject.read()
  fileObject.close()
  return fileContent
filepath="gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"
token_text=fileOpen(filepath)

print(token_text)

print(token_text[:1500])

type(token_text)

import pandas as pd
import numpy as np

print(token_text[0:50])

def imageid_nd_captionsplit(token_text):
  imageid_nd_caption={}
  for each_row in token_text.split('\n'):
    if len(each_row) <20:
      continue
    image_name=each_row.split()[0].split('.')[0]
    image_caption=each_row.split()[1:]
    image_caption=' '.join(image_caption)
    if image_name not in imageid_nd_caption.keys():
      imageid_nd_caption[image_name]=[]
    imageid_nd_caption[image_name].append(image_caption)
  return imageid_nd_caption

imageid_nd_caption=imageid_nd_captionsplit(token_text)

print('Total Rows: '+str(len(imageid_nd_caption)))

imageid_nd_caption['1000268201_693b08cb0e']

imageid_nd_caption['1007320043_627395c3d8']

import string
import copy
def datacleaning(imageid_nd_caption):
  imageid_nd_caption_cleaned=copy.deepcopy(imageid_nd_caption)
  translator = str.maketrans('', '', string.punctuation)
  for key, value in imageid_nd_caption_cleaned.items():
    for each_caption in range(len(value)):
      value[each_caption]=value[each_caption].lower()
      value[each_caption]=value[each_caption].translate(translator)
      value[each_caption]=' '.join( [w for w in value[each_caption].split() if len(w)>1] )
      value[each_caption]=' '.join( [w for w in value[each_caption].split() if w.isalpha()] )
  return imageid_nd_caption_cleaned

imageid_nd_caption_cleaned=datacleaning(imageid_nd_caption)

imageid_nd_caption_cleaned['1000268201_693b08cb0e']

def all_caption_tofile(imageid_nd_caption_cleaned, filepath):
  newfile=open(filepath,'w')
  for key,value in imageid_nd_caption_cleaned.items():
    for each_caption in range(len(value)):
      newfile.write(key+' '+value[each_caption])
      newfile.write('\n')
  newfile.close()

all_caption_tofile(imageid_nd_caption_cleaned,"/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Created/captions.txt")

def totalDistictWords(imageid_nd_caption_cleaned):
  all_words=set()
  for key,value in imageid_nd_caption_cleaned.items():
    for each_caption in range(len(value)):
      all_words.update(value[each_caption].split())
  return all_words

all_words=totalDistictWords(imageid_nd_caption_cleaned)

def load_train_data(filepath):
  traindata=fileOpen(filepath)
  data=[]
  for each_row in traindata.split('\n'):
    if len(each_row)<1:
      continue
    image_id=each_row.split('.')[0]
    data.append(image_id)
  return set(data)

train_data=load_train_data("/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt")

print("Total Train Data : "+str(len(train_data)))

list(train_data)[:50]

test_data=load_train_data("/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt")

print("Total test Data : "+str(len(test_data)))

list(test_data)[:10]

all_images_path="/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Images/"

import glob
images = glob.glob(all_images_path + '*.jpg')

def trainDataFullPath(images, train_data):
  train_data_fullpath=[]
  for i in images:
    if i[len(all_images_path):].split('.')[0] in train_data:
      train_data_fullpath.append(i)
  return train_data_fullpath

train_data_fullpath=trainDataFullPath(images,train_data)

train_data_fullpath[:10]

len(train_data_fullpath)

test_data_fullpath=trainDataFullPath(images,test_data)

test_data_fullpath[:10]

len(test_data_fullpath)

def load_train_datacaption(filepath,train_data):
  captions_file=fileOpen(filepath)
  train_datacaption={}
  for each_row in captions_file.split('\n'):
    if len(each_row)<20:
      continue
    image_id=each_row.split()[0]
    image_caption=each_row.split()[1:]
    image_caption=' '.join(image_caption)
    if image_id in train_data:
      if image_id not in train_datacaption:
        train_datacaption[image_id]=[]
      train_datacaption[image_id].append('startseq ' + image_caption+ ' endseq')
  return train_datacaption

train_datacaption=load_train_datacaption("/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Created/captions.txt",train_data)

len(train_datacaption)

from PIL import Image
from time import time
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image

import warnings
warnings.simplefilter("ignore")

def image_preprocessing(image_path):
    image_after_resize = image.load_img(image_path, target_size=(299, 299))
    image_array = image.img_to_array(image_after_resize)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

inception_model = InceptionV3(weights='imagenet')

new_inception_model = Model(inception_model.input, inception_model.layers[-2].output)

def image_encoding(image):
    image = image_preprocessing(image)
    features_vector = new_inception_model.predict(image)
    features_vector = np.reshape(features_vector, features_vector.shape[1])
    return features_vector

from keras.applications.inception_v3 import preprocess_input
import numpy as np
def strat_image_encoding(train_data_fullpath):
  start_time = time()
  training_image_encoding = {}
  for image in train_data_fullpath:
    training_image_encoding[image[len(all_images_path):]] = image_encoding(image)
  print("Time taken in seconds =", time()-start_time)
  return training_image_encoding

training_image_encoding=strat_image_encoding(train_data_fullpath)

len(training_image_encoding)

training_image_encoding['3661239105_973f8216c4.jpg']

testing_image_encoding=strat_image_encoding(test_data_fullpath)

test_data_fullpath[:10]

testing_image_encoding['396360611_941e5849a3.jpg'].shape

import pickle
from pickle import dump, load
with open('/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Created/train_imagefeatures.pickle', 'wb') as pickler:
    pickle.dump(training_image_encoding, pickler)

with open('/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Created/test_imagefeatures.pickle', 'wb') as pickler:
    pickle.dump(testing_image_encoding, pickler)

list_ofalltrainingcaptions=[]
for key, value in train_datacaption.items():
  for each_caption in value:
    list_ofalltrainingcaptions.append(each_caption)
len(list_ofalltrainingcaptions)

import pickle
from pickle import dump, load
alltrainimages_features=load(open('/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Created/train_imagefeatures.pickle','rb'))

def vocab_reduction(list_ofalltrainingcaptions):
  min_wordcount=10
  eachword_count={}
  for each_caption in list_ofalltrainingcaptions:
    for each_word in each_caption.split(' '):
      eachword_count[each_word]=eachword_count.get(each_word,0)+1
  vocab_afterreduction=[]
  for each_word in eachword_count:
    if eachword_count[each_word]>=min_wordcount:
      vocab_afterreduction.append(each_word)
  return vocab_afterreduction,eachword_count

vocab_afterreduction,eachword_count=vocab_reduction(list_ofalltrainingcaptions)

print("vocabualary reduced from "+str(len(eachword_count))+" to "+str(len(vocab_afterreduction)))

def word_index_dict(vocab_afterreduction):
  word_from_index={}
  index_from_word={}
  index=1;
  for each_word in vocab_afterreduction:
    index_from_word[each_word]=index
    word_from_index[index]=each_word
    index=index+1
  return index_from_word, word_from_index

index_from_word, word_from_index=word_index_dict(vocab_afterreduction)

vocabulary_count=len(word_from_index)+1

vocabulary_count

max_caption=0
for each_caption in list_ofalltrainingcaptions:
  if max_caption<len(each_caption.split()):
    max_caption=len(each_caption.split())

max_caption

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
def caption_predictor(train_datacaption, alltrainimages_features, index_from_word, max_caption, batch_size, vocabulary_count):
  image_list=[]
  caption_inlist=[]
  caption_outlist=[]
  batch_counter=0
  while True:
    for key, value in train_datacaption.items():
      batch_counter=batch_counter+1
      img=alltrainimages_features[key+'.jpg']
      for image_caption in value:
        encoded_caption=[]
        for each_word in image_caption.split(' '):
          if each_word in index_from_word:
            encoded_caption.append(index_from_word[each_word])
        for each_item in range(1,len(encoded_caption)):
          input_sequence=encoded_caption[:each_item]
          output_sequence=encoded_caption[each_item]
          input_sequence=pad_sequences([input_sequence], maxlen=max_caption)[0]
          output_sequence=to_categorical([output_sequence], num_classes=vocabulary_count)[0]
          image_list.append(img)
          caption_inlist.append(input_sequence)
          caption_outlist.append(output_sequence)
      if batch_counter==batch_size:
        yield[[np.array(image_list),np.array(caption_inlist)],np.array(caption_outlist)]
        image_list=[]
        caption_inlist=[]
        caption_outlist=[]
        batch_counter=0

import numpy as np
glove_file=open("/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/glove.6B.200d.txt",encoding="utf-8")
word_embedded_indices={}
for each_row in glove_file:
  word_list=each_row.split()
  each_word=word_list[0]
  weights=np.asarray(word_list[1:],dtype='float32')
  word_embedded_indices[each_word]=weights
glove_file.close()
print('length of word after embedded : '+str(len(word_embedded_indices)))

word_encode_dim=200
word_encode_matrix=np.zeros((vocabulary_count,word_encode_dim))
for word, index in index_from_word.items():
  word_encode_vector=word_embedded_indices.get(word)
  if word_encode_vector is not None:
    word_encode_matrix[index]=word_encode_vector

word_encode_matrix.shape

word_encode_matrix.shape

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras import Input, layers
from keras.layers import LSTM, Embedding,Dense,Dropout,add
image_size_input=Input(shape=(2048,))
image_out_1=Dropout(0.5)(image_size_input)
image_out_2 = Dense(256, activation='relu')(image_out_1)
caption_size = Input(shape=(max_caption,))
caption_out_1 = Embedding(vocabulary_count, word_encode_dim, mask_zero=True)(caption_size)
caption_out_2 = Dropout(0.5)(caption_out_1)
caption_out_3 = LSTM(256)(caption_out_2)
output_decoder_1= add([image_out_2, caption_out_3])
output_decoder_2 = Dense(256, activation='relu')(output_decoder_1)
out = Dense(vocabulary_count, activation='softmax')(output_decoder_2)
training_m = Model(inputs=[image_size_input, caption_size], outputs=out,)

training_m.summary()

training_m.layers

training_m.layers[2].set_weights([word_encode_matrix])
training_m.layers[2].trainable = False

training_m.compile(loss='categorical_crossentropy', optimizer='adam')

epoch_count = 50
batch_size = 900
iterations = len(train_datacaption)//batch_size

for each_epoch in range(epoch_count):
    # call_caption_predictor = caption_predictor(train_datacaption,alltrainimages_features , index_from_word, max_caption, batch_size,vocabulary_count)
    training_m.fit_generator(caption_predictor(train_datacaption,alltrainimages_features , index_from_word, max_caption, batch_size,vocabulary_count), epochs=1, steps_per_epoch=iterations, verbose=1)
    training_m.save('/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/weights_600_after_512/weights_900_2_' + str(each_epoch) + '.h5')

print(keras.backend.eval(training_m.optimizer.lr))

#saving and loading weigths of the model
training_m.load_weights('/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/weights_600_after_512/weights_900_2_49.h5')

test_pickle_file=open('/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Created/test_imagefeatures.pickle','rb')
test_encodedfeatures=pickle.load(test_pickle_file)

print(word_from_index)

import keras.preprocessing
def greedy_search(input_image):
  words='startseq'
  for i in range(max_caption):
    caption_word=[index_from_word[word] for word in words.split() if word in index_from_word]
    print(caption_word)
    caption_sequence=keras.preprocessing.sequence.pad_sequences([caption_word], maxlen=max_caption,padding='post')
    print(caption_sequence)
    predicted_words=training_m.predict([input_image,caption_sequence], verbose=0)
    print(predicted_words[0][-3:])
    predicted_word=np.argmax(predicted_words)
    print(predicted_word)
    max_word=word_from_index[predicted_word]
    words+=' '+max_word
    if max_word=='endseq' or len(words)>max_caption:
      break
  words=words.split()
  words=words[1:-1]
  return ' '.join(words)

import matplotlib.pyplot as plt
from PIL import Image

print([*test_encodedfeatures][0])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image_number=100
input_image=[*test_encodedfeatures][image_number]

image=test_encodedfeatures[input_image].reshape((1,2048))
print(image_number)
pred_img=images[image_number]
print(input_image)


#caption prediction
print(greedy_search(image))
img=mpimg.imread('/content/gdrive/My Drive/Machine Learning/Flickr_Data/Flickr_Data/Images/'+input_image)
imgplot = plt.imshow(img)
k=3
# beam_search(input_image, k)

def prediction_using_beamsearch(input_image,bcount):
  
  words=[[[index_from_word['startseq']],0.0]]
  for i in range(34):
    store_beam_predictions=[]
    for each_word in words:
      print([each_word[0]])
      caption_sequence=keras.preprocessing.sequence.pad_sequences([each_word[0]],maxlen=max_caption,padding='post')
      print(caption_sequence)
      predicted_probs=training_m.predict([input_image,caption_sequence], verbose=0)
      print(predicted_probs)
      predicted_words=np.argsort(predicted_probs[0])[-bcount:]
      print(predicted_words)
      for j in predicted_words:
        succ_word=each_word[0][:]
        succ_prob=each_word[1]
        succ_word.append(j)
        succ_prob=succ_prob+predicted_probs[0][j]
        store_beam_predictions.append([succ_word,succ_prob])
    words=store_beam_predictions
    print(words)
    words = sorted(words,key=lambda x: x[1])
    words=words[-bcount:]
    print(words)
    print("--------------------------------------")
    print()
  words=words[-1][0]
  print(words)
  caption=[]
  for k in words:
    if word_from_index[k]!='endseq':
      caption.append(word_from_index[k])
    else:
      break
  return ' '.join(caption[1:])



print(prediction_using_beamsearch(image,3))

def prediction_using_beamsearch_for_bleu(input_image,bcount):
  
  words=[[[index_from_word['startseq']],0.0]]
  for i in range(34):
    store_beam_predictions=[]
    for each_word in words:
      #print([each_word[0]])
      caption_sequence=keras.preprocessing.sequence.pad_sequences([each_word[0]],maxlen=max_caption,padding='post')
      #print(caption_sequence)
      predicted_probs=training_m.predict([input_image,caption_sequence], verbose=0)
      #print(predicted_probs)
      predicted_words=np.argsort(predicted_probs[0])[-bcount:]
      #print(predicted_words)
      for j in predicted_words:
        succ_word=each_word[0][:]
        succ_prob=each_word[1]
        succ_word.append(j)
        succ_prob=succ_prob+predicted_probs[0][j]
        store_beam_predictions.append([succ_word,succ_prob])
    words=store_beam_predictions
    #print(words)
    words = sorted(words,key=lambda x: x[1])
    words=words[-bcount:]
    #print(words)
    #print("--------------------------------------")
    #print()
  words=words[-1][0]
  #print(words)
  caption=[]
  for k in words:
    if word_from_index[k]!='endseq':
      caption.append(word_from_index[k])
    else:
      break
  return ' '.join(caption[1:])



#BLEU SCORE IMPLEMENTATION
from nltk.translate.bleu_score import sentence_bleu

def blue_score_evaluation(imageid_nd_caption_cleaned, test_encodedfeatures, image_number, image):
  inputimage=[*test_encodedfeatures][image_number]
  inputimage=inputimage.split('.')
  inputimage = inputimage[0]
  actualCaptions = list()
  predictedCaption = prediction_using_beamsearch_for_bleu(image,3)
  predictedCaption = predictedCaption.split(' ')

  for caption in imageid_nd_caption_cleaned[inputimage]:
    tempList = caption.split(' ')
    actualCaptions.append(tempList)
  predictionscore = sentence_bleu(actualCaptions,predictedCaptions)
  return predictionscore

print("BLEU score of the predicted caption: "+str(blue_score_evaluation(imageid_nd_caption_cleaned, test_encodedfeatures, image_number, image)))