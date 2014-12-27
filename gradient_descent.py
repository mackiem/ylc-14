import numpy as np
import pylab as plt
import math
import json
import random
import sys

K = 20

###############################################################################
# Objective function
#
# @param reviews          the review dataset
# @param w                w
# @param b                b
# @param wh               the rough estimate of w by LDA
# @param bh               the rough estimate of b by LDA
# @param mu               the impact of the first regularizer
# @param lambda           the impact of the second regularizer
# @return F
def objFunc(reviews, w, b , wh, bh, mu, lmbd):
    F = 0.0
    for review in reviews:
        F += (np.dot(w[review[0],:], b[review[1],:]) - review[2]) ** 2

    for u in xrange(w.shape[0]):
        F += mu * np.dot(w[u,:] - wh[u,:], w[u,:] - wh[u,:])

    for i in xrange(b.shape[0]):
        F += lmbd * np.dot(b[i,:] - bh[i,:], b[i,:] - bh[i,:])

    return F

###############################################################################
# Gradient Descent
#
# @param reviews          the review dataset
# @param wh               the rough estimate of w by LDA
# @param bh               the rough estimate of b by LDA
# @param maxIterations    the maximum iteration.
# @param stepSize         step size
# @param mu               the impact of the first regularizer
# @param lambda           the impact of the second regularizer
# @return w, b
def GD(reviews, wh, bh, maxIterations, stepSize, mu, lmbd):
    list_L = []
    prevL = 0
    threshold = 0.001

    # initialize w and b
    w = np.copy(wh)
    b = np.copy(bh)
    prevL = objFunc(reviews, w, b, wh, bh, mu, lmbd)

    for iter in xrange(maxIterations):
        dw = np.zeros(w.shape)
        db = np.zeros(b.shape)

        for review in reviews:
            d = np.dot(w[review[0],:], b[review[1],:])
            dw[review[0],:] += (d - review[2]) * b[review[1],:]
            db[review[1],:] += (d - review[2]) * w[review[0],:]

        for u in xrange(w.shape[0]):
            dw[u,:] += lmbd * (w[u,:] - wh[u,:])
        for i in xrange(b.shape[0]):
            db[i,:] += mu * (b[i,:] - bh[i,:])

        # update w and b
        w -= stepSize * dw
        b -= stepSize * db

        ##############################
        L = objFunc(reviews, w, b, wh, bh, mu, lmbd)
        list_L.append(L)

        # When it has converged, stop the iterations.
        if abs(L - prevL) < threshold:
            break
        prevL = L

    return (w, b, list_L)

###############################################################################
# Estimate w and b by LDA
#
# @param filename         review file
# @return w, b
def estimateByLDA(review_filename, lda_result_filename):
    f = open(review_filename)
    dataset = json.load(f)
    f.close()

    user_ids = []
    business_ids = []
    stars = {}
    for review in dataset:
        if review["user_id"] not in user_ids:
            user_ids.append(review["user_id"])
        if review["business_id"] not in business_ids:
            business_ids.append(review["business_id"])

        stars[(review["user_id"], review["business_id"])] = review["stars"]



    f = open(lda_result_filename)
    dataset = json.load(f)
    f.close()

    wh = {}
    bh = {}
    for user_id, u_data in dataset.iteritems():
        theta = np.zeros((0, K))
        for business_id, b_data in u_data.iteritems():
            row = np.zeros((1, K))
            for k,v in b_data["result"]:
                row[0, k] = v
            theta = np.r_[theta, row]


        wh_row = np.sum(theta, axis=0)
        wh_row /= np.sum(wh_row)
        wh[user_id] = wh_row


    for business_id in business_ids:
        total = np.zeros((1, K))
        weight = np.zeros((1, K))
        for user_id, u_data in dataset.iteritems():
            for b_id, b_data in u_data.iteritems():
                if b_id != business_id: continue
                for k,v in b_data["result"]:
                    total[0, k] += v * stars[(user_id, business_id)]
                    weight[0, k] += v

        for k in xrange(K):
            if weight[0, k] > 0.0:
                total[0, k] /= weight[0, k]
            else:
                total[0, k] = 3.0

        bh[business_id] = total

    return wh, bh

###############################################################################
# Mean square error
#
# @param dataset          the test dataset
# @param w                w
# @param b                b
# @return error
def MSE(dataset, w, b):
    error = 0.0
    count = 0
    for data in dataset:
        predict = np.dot(w[data[0],:], b[data[1],:])
        error += (predict - float(data[2])) ** 2
        count += 1

    return error / count

###############################################################################
# create dataset
#   Input: filename
#   Output: return a list of rows.
def createDataset(filename):
    train_data = []
    val_data = []
    test_data = []
    user_ids = []
    business_ids = []

    # ratio of #examples in training/validation/test dataset
    ratio = [0.6, 0.2, 0.2]

    f = open(filename)
    reviews = json.load(f)
    f.close()

    random.seed(0)

    dataset = []
    random.seed(0)
    remained_num_train = len(dataset) * ratio[0]
    remained_num_validation = len(dataset) * ratio[1]
    remained_num_test = len(dataset) * ratio[2]

    for review in reviews:
        u_id = 0
        b_id = 0
        if review["user_id"] in user_ids:
            u_id = user_ids.index(review["user_id"])
        else:
            u_id = len(user_ids)
            user_ids.append(review["user_id"])
        if review["business_id"] in business_ids:
            b_id = business_ids.index(review["business_id"])
        else:
            b_id = len(business_ids)
            business_ids.append(review["business_id"])

        rnd = random.uniform(0, remained_num_train + remained_num_validation + remained_num_test)
        if rnd < remained_num_train:
            train_data.append([u_id, b_id, review["stars"]])
            remained_num_train -= 1
        elif rnd < remained_num_validation:
            val_data.append([u_id, b_id, review["stars"]])
            remained_num_validation -= 1
        else:
            test_data.append([u_id, b_id, review["stars"]])
            remained_num_test -= 1

    return train_data, val_data, test_data, user_ids, business_ids

###############################################################################
# Find the best hyperparameters
def findBestHyperparameters(user_ids, business_ids, train_data, val_data, wh2, bh2):
    wh = np.zeros((len(user_ids), K))
    bh = np.zeros((len(business_ids), K))

    for i in xrange(len(user_ids)):
        wh[i] = wh2[user_ids[i]]
    for i in xrange(len(business_ids)):
        bh[i] = bh2[business_ids[i]]

    stepSizeSet = [0.0001, 0.00005, 0.00002, 0.00001]
    muSet = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]

    min_error = 100000
    for stepSize in stepSizeSet:
        for mu in muSet:
            lmbd = mu
            print("Solving by gradient descent (stepSize=" + str(stepSize) + ", mu=" + str(mu) + ", lambda=" + str(lmbd) + ")")
            w, b, list_L = GD(train_data, wh, bh, maxIterations, stepSize, mu, lmbd)

            print("Performance test on validation data.")
            error = MSE(val_data, w, b)
            print("MSE: " + str(error))
            print("RMSE: " + str(math.sqrt(error)))

            if error < min_error:
                min_error = error
                best_stepSize = stepSize
                best_mu = mu
                best_lmbd = lmbd

    return best_stepSize, best_mu, best_lmbd

###############################################################################
# Show the most important aspect for users according to w
def showMostImportantAspect(w):
    row_sum = np.sum(w, axis=0)

    max_val = 0
    index = -1
    print(row_sum.shape)
    for i in xrange(row_sum.shape[0]):
        if row_sum[i] > max_val:
            max_val = row_sum[i]
            index = i

    print("Most important aspect: " + str(index))

###############################################################################
# Save the results of w and h
def saveResults(user_ids, business_ids, w, b):
    f = open("user_results.txt", "w")
    for i in xrange(len(user_ids)):
        f.write(user_ids[i])
        for k in xrange(K):
            f.write("\t" + str(w[i,k]))
        f.write("\n")
    f.close()

    f = open("business_results.txt", "w")
    for i in xrange(len(business_ids)):
        f.write(business_ids[i])
        for k in xrange(K):
            f.write("\t" + str(b[i,k]))
        f.write("\n")
    f.close()

###############################################################################
# Main function
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("")
        print("Usage: python " + sys.argv[0] + " <review filename> <GD/SC> <LDA result filename> <maxIterations> <stepSize> <mu> <lambda>")
        print("     (e.g. python " + sys.argv[0] + " GD review_3466.json lda_result_new.txt 100000 0.00001 2.0 2.0)")
        print("     (e.g. python " + sys.argv[0] + " SC review_3466.json lda_result_new.txt 100000)")
        print("     (e.g. python " + sys.argv[0] + " LDA review_3466.json lda_result_new.txt 100000)")
        print("")
        sys.exit(1)

    print("<<< Gradient descent >>>")

    option = sys.argv[1]
    review_filename = sys.argv[2]
    lda_result_filename = sys.argv[3]
    maxIterations = int(sys.argv[4])
    stepSize = 10000
    mu = 0.1
    lmbd = 0.1

    if option == "GD":
        stepSize = float(sys.argv[5])
        mu = float(sys.argv[6])
        lmbd = float(sys.argv[7])

    # compute the rough estimate of w and h by LDA
    wh2, bh2 = estimateByLDA(review_filename, lda_result_filename)

    train_data, val_data, test_data, user_ids, business_ids = createDataset(review_filename)

    # find the best hyperparameters
    if option == "SC":
        stepSize, mu, lmbd = findBestHyperparameters(user_ids, business_ids, train_data, val_data, wh2, bh2)

    wh = np.zeros((len(user_ids), K))
    bh = np.zeros((len(business_ids), K))

    for i in xrange(len(user_ids)):
        wh[i] = wh2[user_ids[i]]
    for i in xrange(len(business_ids)):
        bh[i] = bh2[business_ids[i]]

    if option == "LDA":
        print("Performance test on training data.")
        error = MSE(train_data, wh, bh)
        print("MSE: " + str(error))
        print("RMSE: " + str(math.sqrt(error)))

        print("Performance test on validation data.")
        error = MSE(val_data, wh, bh)
        print("MSE: " + str(error))
        print("RMSE: " + str(math.sqrt(error)))

        print("Performance test on test data.")
        error = MSE(test_data, wh, bh)
        print("MSE: " + str(error))
        print("RMSE: " + str(math.sqrt(error)))

        sys.exit()

    print("Solving by gradient descent")
    w, b, list_L = GD(train_data, wh, bh, maxIterations, stepSize, mu, lmbd)

    print("Performance test on training data.")
    error = MSE(train_data, w, b)
    print("MSE: " + str(error))
    print("RMSE: " + str(math.sqrt(error)))

    print("Performance test on validation data.")
    error = MSE(val_data, w, b)
    print("MSE: " + str(error))
    print("RMSE: " + str(math.sqrt(error)))

    print("Performance test on test data.")
    error = MSE(test_data, w, b)
    print("MSE: " + str(error))
    print("RMSE: " + str(math.sqrt(error)))

    showMostImportantAspect(w)

    saveResults(user_ids, business_ids, w, b)


    plt.plot(range(len(list_L)), list_L, "-")
    plt.title("Gradient Descent (stepSize=" + str(stepSize) + ", mu=" + str(mu) + ", labmda=" + str(lmbd) + ")")
    plt.xlim(0, len(list_L))
    plt.savefig("GD_" + str(stepSize) + "_" + str(mu) + "_" + str(lmbd) + ".eps")
    plt.show()

