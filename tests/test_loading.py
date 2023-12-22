import sys
import os
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import re
import PyPDF2
import string
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from unidecode import unidecode

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


# Fungsi untuk membaca teks dari file PDF
# cara 1
# def read_pdf_text(file_path):
#     with open(file_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text

# cara 2
import tika
from tika import parser
def read_pdf_text(file_path):
    parsed = parser.from_file(file_path)
    text = parsed['content']
    return text



# Path file PDF pertama
file1_path = os.path.join(myPath, 'test.pdf')
file1_text = read_pdf_text(file1_path)
print(file1_text)
print()
print()
print()
    

# Preprocessing teks
# [A] Text cleaning
def clean_text(text):
    text = text.lower()                                                 # mengubah semua teks menjadi huruf kecil
    text = re.sub(r'\d+', '', text)                                     # menghapus angka dari teks
    text = text.translate(str.maketrans('', '', string.punctuation))    # menghapus tanda baca dari teks dan mengganti dengan string kosong
    text = re.sub(r'\s+', ' ', text)                                    # menggantikan spasi berulang-ulang dengan satu spasi tunggal
    text = text.strip()                                                 # menghapus spasi yang ada di awal dan akhir teks
    return text

# [B] Tokenization
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)       # memecah teks menjadi kata-kata berdasarkan batasan spasi dan tanda baca. Hasilnya adalah daftar kata-kata yang terpisah.
    return tokens

# [C] Stop word removal
# cara1
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    stop_words.update(list(string.ascii_lowercase))  # Menambahkan semua huruf abjad
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # print(filtered_tokens)
    return filtered_tokens

# cara2
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# factory = StopWordRemoverFactory()
# stopword_remover = factory.create_stop_word_remover()
# def remove_stopwords(tokens):
#     filtered_tokens = [token for token in tokens if token not in stopword_remover.remove(token)]
#     print(filtered_tokens)
#     return filtered_tokens


# # [D] Lemmatization
# tidak udah karena hampir mirip dengan stemming
# def lemmatize_tokens(tokens):
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     return lemmatized_tokens

# [E] Stemming
# def stem_tokens(tokens):
#     stemmer = PorterStemmer()
#     stemmed_tokens = [stemmer.stem(token) for token in tokens]
#     return stemmed_tokens

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
def stem_tokens(tokens):
    stemmer = StemmerFactory().create_stemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


# [F] Removing accents
def remove_accents(text):
    text = unidecode(text)
    return text

# Preprocessing pipeline
def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    tokens = remove_stopwords(tokens)
    # tokens = lemmatize_tokens(tokens)
    tokens = stem_tokens(tokens)
    preprocessed_text = tokens
    # print(tokens)
    preprocessed_text = ' '.join(tokens)
    preprocessed_text = remove_accents(preprocessed_text)
    return preprocessed_text

# Apply preprocessing to your text
preprocessed_text = preprocess_text(file1_text)
corpus = [preprocessed_text, 'okokok brooo wkwk hemm huum', 'yuppp gaes ok ej sak sakk']
tokenized_corpus = [doc.split(" ") for doc in corpus]
print(tokenized_corpus)

# corpus_text = re.sub(r'[^\w\s.]', '', corpus_text)                                    # menghilangkan semua karakter kecuali huruf, angka, spasi, dan titik dari teks
# corpus_lines = corpus_text.split(".")                                                 # membagi teks menjadi beberapa baris berdasarkan tanda titik
# corpus_lines = [line.strip().lower() for line in corpus_lines if line.strip() != ""]  # menghapus spasi di awal dan akhir baris, dikonversi ke huruf kecil, dan menghilangkan baris yang kosong

# # mengubah menjadi daftar kata-kata yang terpisah untuk setiap kalimat
# corpus = corpus_lines
# tokenized_corpus = [doc.split(" ") for doc in corpus]
# print(corpus)

query = "search engine"                                                                # kata yang ingin dicari
tokenized_query = query.lower().split(" ")                                            # mengubah ke huruf kecil dan membagi kalimat menjadi perkata
print(query)

# membuat objek bm25, agar dapat menggunakan metode dan fungsi yang disediakan oleh pustaka rank_bm25 
bm25 = BM25Okapi(tokenized_corpus)                                                    
# menggunakan objek bm25 untuk menghitung skor BM25 (tingkat relevansi antara query dan dokumen)
doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

# mendapatkan n dokumen dengan skor tertinggi yang paling relevan
hasil = bm25.get_top_n(tokenized_query, corpus, n=1)
print(hasil)


algs = [
    BM25Okapi(tokenized_corpus),
    BM25L(tokenized_corpus),
    BM25Plus(tokenized_corpus)
]


def test_corpus_loading():
    for alg in algs:
        assert alg.corpus_size == 3
        assert alg.avgdl == 5
        assert alg.doc_len == [4, 6, 5]


def tokenizer(doc):
    return doc.split(" ")


def test_tokenizer():
    bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
    assert bm25.corpus_size == 3
    assert bm25.avgdl == 5
    assert bm25.doc_len == [4, 6, 5]
