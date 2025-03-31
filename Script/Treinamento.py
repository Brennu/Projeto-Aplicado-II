import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# Bibliotecas para visualização
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas para NLP e modelagem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt_tab')

# Carregar dataset de validação
val=pd.read_csv("https://raw.githubusercontent.com/Brennu/Projeto-Aplicado-II/refs/heads/main/Dataset/twitter_validation.csv", header=None)
# Carregar dataset de treinamento
train=pd.read_csv("https://raw.githubusercontent.com/Brennu/Projeto-Aplicado-II/refs/heads/main/Dataset/twitter_training.csv", header=None)

# Renomeando colunas
train.columns = ['id', 'information', 'type', 'text']
val.columns = ['id', 'information', 'type', 'text']

train_data=train
val_data=val

# Visualizar primeiras linhas
print("Dados de Treino:")
display(train_data.head())
print("\nDados de Validação:")
display(val_data.head())

#Transforma todo o texto para lowercase para padronização
#Isso evita diferenciação entre palavras com caixas diferentes
train_data["lower"]=train_data.text.str.lower()
val_data["lower"]=val_data.text.str.lower()

# Converte todos os valores para string, incluindo números isolados (como '2')
# Isso é necessário pois alguns tweets podem conter apenas números
train_data["lower"]=[str(data) for data in train_data.lower]
val_data["lower"]=[str(data) for data in val_data.lower]

#Remove caracteres especiais, pontuações e símbolos
#Mantém apenas letras, números e espaços
#Importante para tweets que podem conter erros de digitação ou formatação
train_data["lower"]=train_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))
val_data["lower"]=val_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))

#Palavras mais frequentes em tweets positivos incluem termos como "love" e "game", além de outras palavras associadas a sentimentos positivos.
# A diversidade lexical é maior nesta categoria.
word_cloud_text = ''.join(train_data[train_data["type"]=="Positive"].lower)
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Tweets negativos apresentam palavrões com frequência, além de menções a empresas/games específicos como 'facebook' e 'eamaddennfl'.
# Isso pode indicar reclamações direcionadas a essas marcas.
word_cloud_text = ''.join(train_data[train_data["type"]=="Negative"].lower)
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Padrão similar aos tweets negativos, o que pode impactar a performance do modelo,sugerindo possível sobreposição entre essas categorias.
word_cloud_text = ''.join(train_data[train_data["type"]=="Irrelevant"].lower)
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Apresenta perfil lexical distinto, com quase nenhum palavrão e palavras-chave diferentes das outras categorias, indicando maior neutralidade.
word_cloud_text = ''.join(train_data[train_data["type"]=="Neutral"].lower)
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Pré-processamento dos dados
# Agrupamento por marca e tipo para contagem, usando 'id' como referência
plot1 = train.groupby(by=["information","type"]).count().reset_index()
plot1.head()

# Mostra distribuição desigual de sentimentos entre marcas:
# MaddenNFL e NBA2K têm predominância de tweets negativos
# Outras marcas apresentam distribuição mais balanceada
# Neutral é geralmente a categoria mais frequente

plt.figure(figsize=(20,6))
sns.barplot(data=plot1,x="information",y="id",hue="type")
plt.xticks(rotation=90)
plt.xlabel("Marca")
plt.ylabel("Número de Tweets")
plt.grid()
plt.title("Distribuição de Tweets por Marca e Tipo");

# Transforma cada tweet em uma lista de palavras individuais (tokens)
tokens_text = [word_tokenize(str(word)) for word in train_data.lower]

# Achata a lista de listas em uma única lista e calcula elementos únicos
# O tamanho do vocabulário (30,436 tokens) indica alta dimensionalidade, o que pode impactar a performance do modelo
tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Número de tokens únicos: ", len(set(tokens_counter)))

# Demonstra como o texto foi dividido em unidades linguísticas básicas
tokens_text[1]

# N-grams de 1 a 4 palavras (captura frases e contextos)
# Sem remoção de stopwords (pode preservar informações contextuais)
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    ngram_range=(1,4)  # Captura uni-, bi-, tri- e four-grams
)

# Split estratificado (80% treino, 20% teste)
# random_state=0 garante reprodutibilidade
reviews_train, reviews_test = train_test_split(train_data, test_size=0.2, random_state=0)

# Aprende o vocabulário apenas com dados de treino
# Aplica a mesma transformação nos dados de teste
X_train_bow = bow_counts.fit_transform(reviews_train.lower)  # Treino + vocabulário
X_test_bow = bow_counts.transform(reviews_test.lower)

# Formato (n_tweets, n_palavras_únicas) com contagem de ocorrências
X_test_bow

y_train_bow = reviews_train['type']
y_test_bow = reviews_test['type']

#Mostra desbalanceamento moderado com predominância de tweets Negativos e Positivos
y_test_bow.value_counts() / y_test_bow.shape[0]

# C=0.9: Regularização ligeiramente maior
# max_iter=1500: Garantir convergência
model = LogisticRegression(C=0.9, solver="liblinear", max_iter=1500)

model.fit(X_train_bow, y_train_bow)

test_pred = model.predict(X_test_bow)
print("Acurácia: ", accuracy_score(y_test_bow, test_pred) * 100)

y_val_bow = val_data['type']
Val_pred = model.predict(X_val_bow)
print("Acurácia: ", accuracy_score(y_val_bow, Val_pred) * 100)