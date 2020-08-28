#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string


def parseOutText(file):
    """ given an opened email file, parse out all text below the metadata block
        at the top (in Part 2, you will also add stemming capabilities) an
        return a string that contains all the words in the email, separated by
        spaces

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """

    # go back to beginning of the file
    file.seek(0)
    all_text = file.read()

    # split off metadata
    content = all_text.split('X-FileName:')
    words = ''
    if len(content) > 1:
        # remove punctuation
        text_string = content[1].translate(string.maketrans('', ''),
                                           string.punctuation)

        # split the text string into individual words, stem each word, and
        # append the stemmed word to words (make sure there's a single space
        # between each stemmed word)
        stemmer = SnowballStemmer('english')
        split_content = text_string.split()
        words = ' '.join(stemmer.stem(word) for word in split_content)

    return words


def main():
    file_path = '../text_learning/test_email.txt'
    file = open(file_path, 'r')
    text = parseOutText(file)
    file.close()
    print(text)


if __name__ == '__main__':
    main()
