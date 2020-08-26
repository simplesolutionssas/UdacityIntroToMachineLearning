import os
import nltk
from nltk.corpus import stopwords

# check and/or download the nltk data
download_dir = './data'
if os.path.exists(download_dir) is False:
    nltk.download('all', download_dir=download_dir, halt_on_error=False)
else:
    print('nltk data is already present on this folder')

# load the stop words from the nltk sub-folders
nltk.data.path.append(download_dir)
stop_words = stopwords.words('english')

# query the data
print('stop word count: {}'.format(len(stop_words)))
print('first word: {}'.format(stop_words[0]))
print('eleventh word: {}'.format(stop_words[10]))
