import pandas as pd
def term_document_incidence_matrix(data):
    words = set()
    for content in data:
        words.update(content.split())
    words = list(words)
    words.sort()
    matrix = []
    for content in data:
        row = [0]*len(words)
        for word in content.split():
            row[words.index(word)] = 1
        matrix.append(row)
    return pd.DataFrame(matrix,columns=words).T

data = pd.read_csv("data/food_recipes.csv")
matrix = term_document_incidence_matrix(data['directions'][:2000])
matrix