


def embed_to_nod2vec(
    subgraph, dimensions=64, walk_length=30, num_walks=200, workers=4, dim=0
):
    import torch
    from node2vec import Node2Vec

    # Initialize Node2Vec model
    node2vec = Node2Vec(
        subgraph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Retrieve all node embeddings
    all_embeddings = {node: model.wv[node] for node in model.wv.index_to_key}

    # Convert each numpy array to a PyTorch tensor
    tensor_list = [torch.from_numpy(arr) for arr in all_embeddings.values()]

    # Stack the tensors
    embeddings_tensor = torch.stack(tensor_list)

    # Calculate graph embedding
    graph_embedding, _ = torch.max(embeddings_tensor, dim=dim)

    return graph_embedding




def embed_nod2vec_dot_product(
    subgraph, dimensions=64, walk_length=30, num_walks=200, workers=4
):
    import torch
    from node2vec import Node2Vec

    # Initialize Node2Vec model
    node2vec = Node2Vec(
        subgraph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Retrieve all node embeddings
    all_embeddings = {node: model.wv[node] for node in model.wv.index_to_key}

    # Convert each numpy array to a PyTorch tensor
    tensor_list = [torch.from_numpy(arr) for arr in all_embeddings.values()]

    # Stack the tensors
    embeddings_tensor = torch.stack(tensor_list)

    # Calculate the dot product between each pair of embeddings
    dot_products = torch.matmul(embeddings_tensor, embeddings_tensor.T)

    # Extract the upper triangular part excluding the diagonal and convert to one-dimensional vector
    upper_tri_vector = dot_products[
        torch.triu_indices(dot_products.size(0), dot_products.size(1), offset=1)
    ].flatten()

    return dot_products, upper_tri_vector



def pd_get_embeddings(df, column, batch_size=32, as_dataframe=False):
    # Define the model
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModel.from_pretrained('bert-base-uncased')
# embeddings_df = get_embeddings(temp_df, 'lead_contact_title', 512, as_dataframe=True)
# embeddings_df.to_parquet('emb.parquet')
#embeddings_df = pd.read_parquet('emb.parquet')
# Concatenate the original DataFrame with the embeddings DataFrame
#temp_df = pd.concat([temp_df, embeddings_df], axis=1)
#embedding_columns = embeddings_df.columns.tolist()

    # Cast the column to string
    df[column] = df[column].astype(str)

    embeddings_list = []

    # Process the data in chunks
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[i:i+batch_size]
        inputs = tokenizer(list(batch[column]), padding=True, truncation=True, max_length=128, return_tensors='pt')

        # Get the embeddings
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state

        # Compute mean of all token embeddings for each sentence to create sentence embeddings
        sentence_embeddings = embeddings.mean(dim=1)

        # Convert tensor to numpy array and append to list
        numpy_embeddings = sentence_embeddings.numpy()
        embeddings_list.append(numpy_embeddings)
    
    # Concatenate all the chunks
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    
    if as_dataframe:
        num_cols = all_embeddings.shape[1]
        cols = [f'{column}_embed_{i}' for i in range(num_cols)]
        df_embeddings = pd.DataFrame(all_embeddings, columns=cols)
        return df_embeddings

    return all_embeddings


