from bs4 import BeautifulSoup

# Example sentence (normally this would come from a dataset)
sentence = "This is a sample sentence about machine learning!"

# Create a list to store processed sentences
sentences = []

soup = BeautifulSoup(sentence, features="html.parser")
sentence = soup.get_text()
# More would be here in an acutal program
stopwords = ['a', 'about', 'above', 'yours', 'yourself', 'yourselves']

words = sentence.split()
filtered_sentence = ''
for word in words:
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + ' '
    sentences.append(filtered_sentence)
# Adds a way to remove punctuation
import string
table = str.maketrans('', '', string.punctuation)
words = sentence.split()
filtered_sentence = ''
for word in words:
    word = word.translate(table)
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + ' '
sentences.append(filtered_sentence)
