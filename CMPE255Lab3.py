
# coding: utf-8

# In[2]:


matplotlib inline


# In[35]:


def load_data():
    data = []
    data_labels = []
    with open("/Users/mallampati/Downloads/pos_tweets.txt") as f:
        for i in f: 
            data.append(i) 
            data_labels.append('pos')

    with open("/Users/mallampati/Downloads/neg_tweets.txt") as f:
        for i in f: 
            data.append(i)
            data_labels.append('neg')

    return data, data_labels

def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()
    return features_nd

def train_then_build_model(data, data_labels, features_nd):
    from sklearn.cross_validation import train_test_split
    # set training % to 80%.
    X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80, 
        random_state=1234)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(X_test)

    # TODO
    # print first 10 predictions in this format ::{prediction}::{tweet}
    for i in range(10):
        print("prediction={}".format(y_pred[i]))
        # features_nd.tolist().index(X_test[i].tolist())
        print("tweet={}".format(data[features_nd.tolist().index(X_test[i].tolist())]))
        
    
    # TODO
    # print accuracy
    from sklearn.metrics import accuracy_score
    print("Accuracy={}".format(accuracy_score(y_test, y_pred)))

def process():
    data, data_labels = load_data()
    features_nd = transform_to_features(data)
    train_then_build_model(data, data_labels, features_nd)


process()

