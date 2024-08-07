def recursive_raptor(embeddings, summaries, model, level=1, max_levels=3):
    if level > max_levels or len(embeddings) <= 1:
        return summaries

    n_clusters = min(5, len(embeddings))
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(embeddings)
    probabilities = gmm.predict_proba(embeddings)

    cluster_texts = [[] for _ in range(n_clusters)]
    for i, embedding in enumerate(embeddings):
        cluster_idx = np.argmax(probabilities[i])
        cluster_texts[cluster_idx].append(i)

    new_embeddings = []
    new_summaries = []
    for indices in cluster_texts:
        if indices:
            texts = [summaries[i] for i in indices]
            summary = summarize(texts)
            new_summaries.append(summary)
            new_embedding = model.encode([summary])[0]
            new_embeddings.append(new_embedding)

    return new_summaries

hierarchical_summaries = recursive_raptor(summary_embeddings, summaries, model)
print(hierarchical_summaries)