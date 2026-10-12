"""
truthguard/clustering.py - K-Means topic discovery on TF-IDF features.
Clusters are unsupervised groupings of similar wording, not verified categories.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import config
from truthguard.modeling import clean_text

NOTICE = ("Clusters are automatic groupings of similar wording. They are not "
          "verified news categories and may change with different input.")


class ClusteringError(ValueError):
    """Raised when the texts cannot be clustered (too few or too similar)."""


def cluster_texts(texts, n_clusters):
    if len(set(texts)) < n_clusters:
        raise ClusteringError(
            "Not enough distinct texts for that many clusters. "
            "Add more different texts or choose fewer clusters.")
    vectorizer = TfidfVectorizer(preprocessor=clean_text, stop_words="english")
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        raise ClusteringError("The texts contain no usable words after cleaning.")

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=config.RANDOM_SEED)
    labels = km.fit_predict(X)
    terms = np.array(vectorizer.get_feature_names_out())
    distances = km.transform(X)   # distance of every text to every centroid

    clusters = []
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        top_terms = terms[km.cluster_centers_[c].argsort()[::-1][:5]].tolist()
        best = members[distances[members, c].argmin()]   # closest to the centroid
        example = texts[best]
        clusters.append({
            "cluster": c, "size": int(len(members)), "top_terms": top_terms,
            "example": example[:config.SAVED_EXCERPT_LENGTH],
        })

    silhouette = None
    if 1 < len(set(labels)) < len(texts):
        silhouette = round(float(silhouette_score(X, labels)), 3)

    points = None   # 2-D picture of the clusters (optional)
    if len(texts) >= 3 and X.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=config.RANDOM_SEED).fit_transform(X.toarray())
        points = [{"x": round(float(x), 4), "y": round(float(y), 4), "cluster": int(l)}
                  for (x, y), l in zip(coords, labels)]

    return {"n_clusters": n_clusters, "n_texts": len(texts), "clusters": clusters,
            "silhouette": silhouette, "points": points, "notice": NOTICE}
