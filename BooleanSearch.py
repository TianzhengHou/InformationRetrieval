from sklearn.feature_extraction.text import CountVectorizer

# Useing library is way more faster than implement from the start
def lib_term_document_incidence_matrix(data):
    text = data['directions']
    vectorizer = CountVectorizer(lowercase=True)
    X = vectorizer.fit_transform(text)
    term_document_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    return term_document_matrix.T

def boolean_search(data, matrix, search_terms, operator='AND'):
    try:
        # Ensure that the search terms are in lowercase
        search_terms = [term.lower() for term in search_terms]
                
        # Filter the matrix to include only the search terms
        filtered_matrix = matrix.loc[search_terms]
    
        # Find all reviews where all terms appear (boolean AND operation)
        if operator == 'AND':
            valid_indices = filtered_matrix.columns[(filtered_matrix > 0).all()]
            
        # Find all reviews where any term appears (boolean OR operation)
        elif operator == 'OR':
            valid_indices = filtered_matrix.columns[(filtered_matrix > 0).any()]
            
        else:
            raise ValueError("Operator must be 'AND' or 'OR'")

        # Select and return the relevant data using the valid indices
        return data.loc[valid_indices, ['title', 'directions']]

    except KeyError as e:

        # Handle the case where one or more search terms are not in the matrix
        print(f"Warning: {str(e).strip('[]')} not found")
        return pd.DataFrame()  # Return an empty DataFrame if any term is not found
    
matrix = lib_term_document_incidence_matrix(data)
result = boolean_search(data, matrix, ['brown','sugar'], operator='OR')
print(result)