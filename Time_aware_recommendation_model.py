import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


'''-------------------article popularity prediction model----------------'''
class news_popularity_pred(nn.Module):
    def __init__(self, content_embedding_dim = 100, hidden_size = (32,16), dropout_prob = 0.5, epochs = 30, batchsize = 16, lr = 0.0003):
        super(news_popularity_pred, self).__init__()
        self.content_embedding_dim = content_embedding_dim
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(self.content_embedding_dim + 1, self.hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.hidden_size[1], 1)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
    
    def article_id_map(self, article_dt):
        ids = article_dt['articleID'].unique()
        id_to_idx = {id_text: idx for idx, id_text in enumerate(ids)}
        return id_to_idx
    
    def text_to_w2v_embedding(self, text, model):
        words = re.split(r'[^\w]', text)
        words = [word for word in words if word in model.wv]
        if not words:
            return np.zeros(self.content_embedding_dim)
        return np.mean([model.wv[word] for word in words], axis = 0)
    
    def content_embedding(self, article_dt):
        word2vec_model = Word2Vec.load('word2vec_model.model')
        content_emb = article_dt['text'].apply(lambda x: self.text_to_w2v_embedding(x, word2vec_model))
        content_embedding = np.vstack(content_emb)
        m = np.mean(content_embedding, axis = 0)
        s = np.std(content_embedding, axis = 0)
        content_embedding = (content_embedding - m) / s
        return content_embedding
    
    def recency_embedding(self, article_dt, cur_time):
        cur_time = pd.to_datetime(cur_time)
        recencies = (cur_time - pd.to_datetime(article_dt['pub_date']).dt.tz_localize(None)).dt.total_seconds() / 3600
        recencies = recencies.to_numpy()
        recencies = (recencies - min(recencies)) / (max(recencies) - min(recencies))
        return recencies
    
    def content_recency_embedding(self, article_dt, cur_time):
        content = self.content_embedding(article_dt)
        recency = self.recency_embedding(article_dt, cur_time)
        combined = np.concatenate((content, recency.reshape((-1,1))), axis = 1)
        
        return combined
    
    def total_comment(self, article_comment_dt, cur_time, articleid):
        comment_till_cur = article_comment_dt[article_comment_dt['createDate'] <= cur_time]
        comment_cnt = comment_till_cur.groupby('articleID').size().reset_index(name = 'comment_count')
        comment_cnt = comment_cnt.set_index('articleID').reindex(article_comment_dt['articleID'].unique(), fill_value=0)
        return comment_cnt.loc[articleid,:].item()
    
    def article_popularity(self, article_comment_dt, cur_time):
        articles = article_comment_dt['articleID'].unique()
        comment_till_cur = article_comment_dt[article_comment_dt['createDate'] <= cur_time]
        comment_cnt = comment_till_cur.groupby('articleID').size().reset_index(name = 'comment_count')
        comment_cnt = comment_cnt.set_index('articleID').reindex(article_comment_dt['articleID'].unique(), fill_value=0)
        popularities = pd.DataFrame({'articleID': articles})
        popularities = popularities.merge(comment_cnt, on = 'articleID', how = 'left').fillna(0)
        total = popularities['comment_count'].sum()
        popularities['popularity'] = popularities['comment_count'] / total
        popularities['popularity'] = (popularities['popularity'] - min(popularities['popularity'])) / (max(popularities['popularity']) - min(popularities['popularity']))
        return popularities['popularity'].to_numpy()
        
    def forward(self, x):
        output = self.net(x)
        return output
    
    def model_training(self, train_x, train_y):
        train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
        train_loader = DataLoader(train_dataset, batch_size = self.batchsize, shuffle = True)

        avg_loss_lst = []
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0

            for x, y in train_loader:
                self.optimizer.zero_grad()
                pred = self.forward(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}")
            avg_loss_lst.append(avg_loss)
        return avg_loss_lst

    def model_evaluation(self, val_x, val_y):
        val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y))
        val_loader = DataLoader(val_dataset, batch_size = self.batchsize)
        
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for x,y in val_loader:
                pred = self.forward(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"Evaluation Loss: {avg_loss}")



'''----------------user interest embedding model---------------'''
class user_based_embedding(nn.Module):
    def __init__(self, hidden_dim = 64, output_dim = 64, content_embedding_dim = 100):
        super(user_based_embedding, self).__init__()
        self.content_embedding_dim = content_embedding_dim
        self.lstm = nn.LSTM(self.content_embedding_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def user_commented_article(self, article_comment_dt, userid, cur_time):
        current_time = pd.to_datetime(cur_time)
        commented_article = article_comment_dt[article_comment_dt['createDate'] <= cur_time]
        user_commented = commented_article[commented_article['userID'] == userid]
        return user_commented
    
    def user_commented_article_sequence(self, group):
        commented_sequence = group.sort_values('createDate')['articleID'].to_list()
        return commented_sequence
    
    def text_to_w2v_embedding(self, text, model):
        words = re.split(r'[^\w]', text)
        words = [word for word in words if word in model.wv]
        if not words:
            return np.zeros(self.content_embedding_dim)
        return np.mean([model.wv[word] for word in words], axis = 0)
    
    def content_embedding(self, article_dt):
        word2vec_model = Word2Vec.load('word2vec_model.model')
        content_emb = article_dt[['articleID','text']]
        content_emb['embedding'] = content_emb['text'].apply(lambda x: self.text_to_w2v_embedding(x, word2vec_model))
        article_emb_dict = dict(zip(content_emb['articleID'], content_emb['embedding']))
        return article_emb_dict
    
    def user_commented_article_embedding(self, user_comment_dt):
        word2vec_model = Word2Vec.load('word2vec_model.model')
        return np.vstack(user_comment_dt.apply(lambda row: self.text_to_w2v_embedding(row['text'], word2vec_model), axis = 1))
    
    def all_user_content_embedding(self, article_dt, article_comment_dt, cur_time):
        sorted_comment_dt = article_comment_dt[article_comment_dt['createDate'] <= cur_time].sort_values(['createDate'])
        article_emb_dict = self.content_embedding(article_dt)
        print('Start embedding')
        print('Extracting commented article sequence')
        user_content_embedding = sorted_comment_dt.groupby('userID').apply(self.user_commented_article_sequence).reset_index()
        user_content_embedding.columns = ['userID', 'commented_article_sequence']
        
        print('Embedding')
        user_content_embedding['embedding'] = user_content_embedding['commented_article_sequence'].apply(lambda x: np.stack([article_emb_dict[article] for article in x]))
        all_user_emb = dict(zip(user_content_embedding['userID'], user_content_embedding['embedding']))
        print('Embedding completed')
        return all_user_emb

    def embedding_padding(self, all_user_emb):
        content_embeddings = [v for v in all_user_emb.values()]
        lengths = [v.shape[0] for v in all_user_emb.values()]
        sorted_len, sorted_idx = torch.sort(torch.tensor(lengths), descending = True)
        userids = list(all_user_emb.keys())
        userids = [userids[i] for i in sorted_idx.tolist()]
        content_sorted_tensors = [torch.from_numpy(content_embeddings[i][::-1].copy()) for i in sorted_idx]
        print('Start padding sequence')
        padded_input = nn.utils.rnn.pad_sequence(content_sorted_tensors, batch_first = True)
        return userids, padded_input
    
    def forward(self, x):
        output_, (hidden, cell) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output


'''----------------recommendation model------------------'''
class news_article_recommendation(nn.Module):
    def __init__(self, lstm_hidden_dim = 100, lstm_output_dim = 100):
        super(news_article_recommendation, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.lstm_model = user_based_embedding(self.lstm_hidden_dim, self.lstm_output_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.lstm_model.parameters(), lr = 0.0001)
        self.epochs = 5
        
    def get_true_article(self, article_dt, article_comment_dt, userid):
        true_article_ids = article_comment_dt[article_comment_dt['userID'] == userid]['articleID'].unique()
        articles = article_dt.reset_index()
        true_article_position = articles.loc[articles['articleID'].isin(true_article_ids)].index
        return true_article_ids, true_article_position
    
    def label(self, idx, n):
        label = np.zeros(n)
        label[idx] = 1
        return label
    
    def get_true_label(self, article_dt, article_comment_dt, userid):
        article_num = article_dt['articleID'].nunique()
        labels = np.zeros(article_num)
        true_article_ids, true_article_position = self.get_true_article(article_dt, article_comment_dt, userid)
        labels[true_article_position] = 1
        return labels
    
    def true_label(self, users, article_dt, article_comment_dt):
        articles = article_dt.reset_index()
        n_articles = article_dt['articleID'].nunique()
        user_article_lists = article_comment_dt.groupby('userID')['articleID'].agg(list).reset_index()
        user_article_lists['article_pos'] = user_article_lists['articleID'].apply(lambda x: articles.loc[articles['articleID'].isin(x)].index)
        user_article_lists['label'] = user_article_lists['article_pos'].apply(lambda x: self.label(x, n_articles))
        sorting_key = {k:v for k,v in enumerate(users)}
        user_article_lists['sorting_key'] = user_article_lists['userID'].map(sorting_key)
        user_article_sorted = user_article_lists.sort_values(by = 'sorting_key')
        
        true_label = np.vstack(user_article_sorted['label'])
        return user_article_sorted, true_label
    
    def user_embed(self, article_dt, article_comment_dt, cur_time):
        all_user_embed = self.lstm_model.all_user_content_embedding(article_dt, article_comment_dt, cur_time)
        user_ids, all_user_embed_padded = self.lstm_model.embedding_padding(all_user_embed)
        return user_ids, all_user_embed_padded
        
    def user_embed_lstm(self, all_user_embed_padded):
        user_embed_lstm = self.lstm_model.forward(all_user_embed_padded)
        user_embed_normalized = F.normalize(user_embed_lstm, p = 2, dim = 0).float()
        return user_embed_normalized
    
    def user_article_sim(self, user_embed, article_embed):
        article_embed_normalized = F.normalize(torch.tensor(article_embed), p = 2, dim = 0).float()
        article_user_sim_mat = torch.matmul(user_embed, article_embed_normalized.t())
        return article_user_sim_mat
    
    def recommend(self, existing_user, user, user_embed, article_embed, article_user_sim_mat, article_dt, article_comment_dt, article_popularity, k):
        articles = article_dt.reset_index()
        if user in existing_user:
            recommend_rating = 0.5 * article_user_sim_mat + 0.5 * article_popularity.t()
            topk_val, topk_idx = torch.topk(recommend_rating, k = k, dim = 1)
            user_topk_idx = topk_idx.tolist()[0]
            user_topk_article = articles.loc[user_topk_idx, 'articleID']
            return user_topk_article.tolist()
        else:
            topk_val, topk_idx = torch.topk(article_popularity.t(), k = k, dim = 1)
            topk_idx = topk_idx.tolist()[0]
            return articles.loc[topk_idx, 'articleID'].tolist()
    
    def recommend_label(self, i, j, article_user_sim_mat, article_dt, article_popularity, k):
        gumbel_softmax = F.gumbel_softmax(article_user_sim_mat, hard = False)
        topk_indices = torch.topk(gumbel_softmax, k = k, dim = 1).indices
        one_hot_topk = F.one_hot(topk_indices, num_classes = article_user_sim_mat.size(1))
        differentiable_topk = one_hot_topk.sum(dim = 1)
        recommend_rating = differentiable_topk
        return recommend_rating
    
    def recommend_label2(self, users, i, j, article_user_sim_mat, article_dt, article_popularity, k):
        if user in users:
            gumbel_softmax = F.gumbel_softmax(0.5 * article_user_sim_mat + 0.5 * article_popularity_pred.t()[i:j], hard = False)
            topk_indices = torch.topk(gumbel_softmax, k = k, dim = 1).indices
            one_hot_topk = F.one_hot(topk_indices, num_classes = article_user_sim_mat.size(1))
            differentiable_topk = one_hot_topk.sum(dim = 1)
            recommend_res = differentiable_topk
            return recommend_res
        else:
            topk_indices = torch.topk(article_popularity_pred.t(), k = k, dim = 1).indices
            one_hot_topk = F.one_hot(topk_indices, num_classes = 2)
            return one_hot_topk
    
    def model_training(self, userids, train_x, train_y, article_embed, article_dt, article_popularity, k):
        for param in self.lstm_model.parameters():
            param.requires_grad_(True)
        
        batch_size = 128
        avg_loss_lst = []
        
        for epoch in range(self.epochs):
            self.lstm_model.train()
            total_loss = 0
            correct_predictions = 0
            
            batch = 0
            for i in range(0, train_x.shape[0], batch_size):
                batch += 1
                self.optimizer.zero_grad()
                x = train_x[i:i + batch_size]
                userid = userids[i:i + batch_size]
                y = train_y[i:i + batch_size]
                
                user_embed = self.user_embed_lstm(x)
                article_user_sim_mat = self.user_article_sim(user_embed, article_embed)
                pred = self.recommend_label(i, i + batch_size, article_user_sim_mat, article_dt, article_popularity, k)
                loss = self.criterion(torch.sigmoid(pred), y)
                loss.backward(retain_graph = True)
                
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / (train_x.shape[0] // batch_size)
        

    def model_pred(self, existing_users, userids, val_x, val_y, article_embed, article_dt, article_comment_dt, article_popularity, k):
        
        self.lstm_model.eval()
        pred_df = pd.DataFrame({'userID': userids})
        preds = []
            
        for i in range(0, val_x.shape[0]):
            x = val_x[i]
            userid = userids[i]
            user_embed = self.user_embed_lstm(x)
            article_user_sim_mat = self.user_article_sim(user_embed, article_embed)
            pred = self.recommend(existing_users, userid, user_embed, article_embed, article_user_sim_mat, article_dt, article_comment_dt, article_popularity, k)
            preds.append(pred)
        
        pred_df['articles'] = preds
        return pred_df



