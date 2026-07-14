import csv
import string
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# More stopwords would be here in a deployed program
stopwords = ['a', 'about', 'above', 'yours', 'yourself', 'yourselves', 'is']

table = str.maketrans('', '', string.punctuation)

sentences=[]
labels=[]
with open('binary.csv', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        labels.append(int(row[0]))
        sentence = row[1]
        # Strip out any HTML tags so we're left with just plain text
        soup = BeautifulSoup(sentence, "html.parser")
        sentence = soup.get_text()
        sentence = sentence.replace(',', ' , ')
        sentence = sentence.replace('.', ' . ')
        sentence = sentence.replace('-', ' - ')
        sentence = sentence.replace('/', ' / ')
        words = sentence.split()
        filtered_sentence = ''
        for word in words:
            word = word.translate(table)
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + ' '
        sentences.append(filtered_sentence)
# Puts 28000 into training and the rest into testing
training_size = 28000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
# Creates word index from training set
# Uses tokenizer to create a vocabulary w/ up to 20000 words
vocab_size = 20000
max_length = 10 # 10 words per sentence max
trunc_type = 'post' # Cuts off at the end
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words= vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_padded = pad_sequences(training_sequences,
                                maxlen = max_length,
                                padding = padding_type,
                                truncating=trunc_type
)
# Inspects results
print(training_sequences[0])
print(training_padded[0])