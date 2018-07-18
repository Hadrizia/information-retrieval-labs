from surprise import KNNWithMeans, SVD
from surprise import Dataset, accuracy
from surprise.model_selection import cross_validate, train_test_split

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

#Spliting data in train (85%) and test (15%)
trainset, testset = train_test_split(data, test_size=.15)


# Use the famous SVD algorithm.
#algo = SVD()

# Using KNN with k = 40
algo = KNNWithMeans(k=40, sim_options={'name': 'pearson_baseline', 'user_based': True})

# Run 10-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)

# Retrieve the trainset.
#trainset = data.build_full_trainset()

# Train the model
algo.fit(trainset)

userid = str(196)
itemid = str(302)
actual_rating = 4

# Getting prediction from userId = 196, itemId = 302 and actual_rating = 4
# TODO: Run this for all itens consumed by userID from test and sort
pred = algo.predict(userid, itemid)
print(testset)
print (pred)

# run the trained model against the testset
test_pred = algo.test(testset)

# get RMSE
#print("User-based Model : Test Set")
print("Test Set")
accuracy.rmse(test_pred, verbose=True)

# if you wanted to evaluate on the trainset
print("Training Set")
train_pred = algo.test(trainset.build_testset())
accuracy.rmse(train_pred)
