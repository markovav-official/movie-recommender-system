import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AVAILABLE_MODELS = ['nearest_neighbors', 'lightgcn', 'all']


def evaluate_nearest_neighbors():
    # load model
    model_path = '../models/nearest_neighbors.pkl'
    preprocessor_path = '../models/nearest_neighbors_preprocessor.pkl'
    model: NearestNeighbors = pickle.load(open(model_path, 'rb'))
    preprocessor: ColumnTransformer = pickle.load(
        open(preprocessor_path, 'rb'))

    # Load users data
    users_path = './data/u.user'
    users_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv(users_path, sep='|',
                        names=users_columns, encoding='latin-1')

    # Load movies data
    movies_path = './data/u.item'
    movies_columns = ['movie_id', 'title', 'release_date',
                      'video_release_date', 'IMDb_URL'] + ['genre_' + str(i) for i in range(19)]
    movies = pd.read_csv(movies_path, sep='|', names=movies_columns,
                         encoding='latin-1', usecols=range(24))

    # Load ratings data
    ratings_path = './data/u.data'
    raings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(ratings_path, sep='\t', names=raings_columns)

    # Combine datasets
    data = pd.merge(pd.merge(ratings, users), movies)

    # Select relevant features
    features = data[['user_id', 'age', 'gender',
                     'occupation', 'zip_code', 'movie_id', 'rating']]
    labels = data[['user_id', 'movie_id', 'rating']]

    # preprocess data
    X = preprocessor.transform(features.drop(
        columns=['user_id', 'movie_id', 'rating']))
    y = labels

    rating_threshold = 3
    k = 10

    precisions = []
    recalls = []

    def precision_recall_at_k(model, X, y, k=5, rating_threshold=4.0):
        precisions = []
        recalls = []

        for user_id in np.unique(y['user_id']):
            # Get the indices and ratings for the user
            user_indices = np.where(y['user_id'].values == user_id)[0]
            user_ratings = y.iloc[user_indices]

            # Find movies rated above the threshold
            relevant_movies = set(
                user_ratings[user_ratings['rating'] >= rating_threshold]['movie_id'])

            # Skip users with fewer than k relevant movies
            if len(relevant_movies) < k:
                continue

            # Get recommendations
            user_data = X[user_indices, :]
            distances, indices = model.kneighbors(user_data, n_neighbors=k)
            recommended_movies = set(y.iloc[indices.flatten()]['movie_id'])

            # Calculate precision and recall
            n_relevant_and_recommended = len(
                relevant_movies.intersection(recommended_movies))
            precision = n_relevant_and_recommended / k
            recall = n_relevant_and_recommended / len(relevant_movies)

            precisions.append(precision)
            recalls.append(recall)

        # Calculate average precision and recall across all users
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0

        return avg_precision, avg_recall

    # Evaluate the model using P@k and R@k with rating threshold 3
    precision, recall = precision_recall_at_k(
        model, X, y, k=10, rating_threshold=3.0)
    print(f"Precision@10 (rating_threshold=3): {precision}")
    print(f"Recall@10 (rating_threshold=3): {recall}")

    print()

    # Evaluate the model using P@k and R@k with rating threshold 4
    precision, recall = precision_recall_at_k(
        model, X, y, k=10, rating_threshold=4.0)
    print(f"Precision@10 (rating_threshold=4): {precision}")
    print(f"Recall@10 (rating_threshold=4): {recall}")


class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class RecSysGNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        num_users,
        num_movies
    ):
        super(RecSysGNN, self).__init__()

        self.model = LightGCNConv()
        self.embedding = nn.Embedding(num_users + num_movies, latent_dim)
        self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))

        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        embs = [emb0]

        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        out = torch.mean(torch.stack(embs, dim=0), dim=0)

        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items]
        )


def evaluate_lightgcn():
    # Load ratings data
    ratings_path = './data/u.data'
    ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv(ratings_path, sep='\t', names=ratings_columns, encoding='latin-1')
    df = df[df['rating'] >= 3]

    train, test = train_test_split(df.values, test_size=0.2, random_state=RANDOM_SEED)
    train_df = pd.DataFrame(train, columns=df.columns)
    test_df = pd.DataFrame(test, columns=df.columns)

    le_user = LabelEncoder()
    le_item = LabelEncoder()

    train_df['user_id_idx'] = le_user.fit_transform(train_df['user_id'].values)

    train_df['movie_id_idx'] = le_item.fit_transform(train_df['movie_id'].values)

    train_user_ids = train_df['user_id'].unique()
    train_movie_ids = train_df['movie_id'].unique()

    test_df = test_df[
        (test_df['user_id'].isin(train_user_ids)) &
        (test_df['movie_id'].isin(train_movie_ids))
    ]

    test_df['user_id_idx'] = le_user.transform(test_df['user_id'].values)
    test_df['movie_id_idx'] = le_item.transform(test_df['movie_id'].values)

    n_users = train_df["user_id_idx"].nunique()
    n_movies = train_df["movie_id_idx"].nunique()

    u_t = torch.LongTensor(train_df.user_id_idx)
    i_t = torch.LongTensor(train_df.movie_id_idx) + n_users

    train_edge_index = torch.stack((
        torch.cat([u_t, i_t]),
        torch.cat([i_t, u_t])
    )).to(device)

    def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_movies, test_data, K):
        # compute the score of all user-item pairs
        relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))

        # create dense tensor of all user-item interactions
        i = torch.stack((
            torch.LongTensor(train_df['user_id_idx'].values),
            torch.LongTensor(train_df['movie_id_idx'].values)
        ))
        v = torch.ones((len(train_df)), dtype=torch.float64)
        interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_movies), device=device).to_dense()

        # mask out training user-item interactions from metric computation
        relevance_score = torch.mul(relevance_score, (1 - interactions_t))

        # compute top scoring items for each user
        topk_relevance_indices = torch.topk(relevance_score, K).indices
        topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(), columns=['top_indx_'+str(x+1) for x in range(K)])
        topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
        topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
        topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]

        # measure overlap between recommended (top-scoring) and held-out user-item interactions
        test_interacted_items = test_data.groupby('user_id_idx')['movie_id_idx'].apply(list).reset_index()
        metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id_idx', right_on=['user_ID'])
        metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.movie_id_idx, metrics_df.top_rlvnt_itm)]

        metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm'])/len(x['movie_id_idx']), axis=1)
        metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm'])/K, axis=1)

        return metrics_df['recall'].mean(), metrics_df['precision'].mean()

    K = 10
    latent_dim = 64
    n_layers = 3

    # Load model
    model_path = '../models/light-gcn.pt'

    model = RecSysGNN(
        latent_dim=latent_dim,
        num_layers=n_layers,
        num_users=n_users,
        num_movies=n_movies
    ).to(device)

    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        _, out = model(train_edge_index)
        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_movies))
        test_topK_recall,  test_topK_precision = get_metrics(
            final_user_Embed, final_item_Embed, n_users, n_movies, test_df, K
        )

    print(f"Precision@10: {test_topK_precision}")
    print(f"Recall@10: {test_topK_recall}")


def main():
    # read args (model name)
    if len(sys.argv) < 2:
        print('Please specify model name, available models are {}'.format(
            ', '.join(list(map(lambda x: f"'{x}'", AVAILABLE_MODELS)))))
        return
    model_name = sys.argv[1]

    # check model name
    if model_name not in AVAILABLE_MODELS:
        print('Invalid model name, please choose from {}'.format(
            ', '.join(list(map(lambda x: f"'{x}'", AVAILABLE_MODELS)))))
        return

    if model_name == 'nearest_neighbors' or model_name == 'all':
        print('Nearest Neighbors')
        evaluate_nearest_neighbors()

    if model_name == 'all':
        print('\n====================================\n')

    if model_name == 'lightgcn' or model_name == 'all':
        print('LightGCN')
        evaluate_lightgcn()


if __name__ == '__main__':
    main()
