import pandas as pd

def map_function(index,row):
    doc_id = index
    content = row["directions"]
    # Split directions into tokens (words)
    tokens = re.findall(r'\b\w+\b', content.lower())
    return [(token, doc_id) for token in tokens]

def reduce_function(shuffled_data):
    inverted_index = {}
    for token, doc_id in shuffled_data:
        if token not in inverted_index:
            inverted_index[token] = []
        inverted_index[token].append(doc_id)
    return inverted_index

def create_inverted_index_map_reduce(data):
    mapped_data = [map_function(i,row) for i, row in data.iterrows()]
    
    reduced_index = reduce_function([item for sublist in mapped_data for item in sublist])
    return reduced_index

def inverted_index_search(recipe_data, inverted_index, search_terms):
    try:
        # Find all indices where all terms appear (boolean AND operation)
        sets_of_indices = [set(inverted_index[term]) for term in search_terms if term in inverted_index]
        if not sets_of_indices:
            return pd.DataFrame()  # Return an empty DataFrame if no indices found

        # Intersect all sets of indices to find common ones
        valid_indices = set.intersection(*sets_of_indices)

        # Convert the set of valid indices to a list
        valid_indices_list = list(valid_indices)

        # Select and return the relevant recipes using the valid indices
        return recipe_data.loc[valid_indices_list, ['title', 'directions']]
    except KeyError as e:
        # Handle the case where one or more search terms are not in the matrix
        print(f"Warning: {str(e).strip('[]')} not found in recipe terms.")
        return pd.DataFrame()  # Return an empty DataFrame if any term is not found
    
inverted_index = create_inverted_index_map_reduce(data)
result = inverted_index_search(data, inverted_index, ['brown','suger'])
print(result)
print(f'\n Returned {len(result)} documents')