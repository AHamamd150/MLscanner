import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
# from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def obs1(x1, x2):
    p1 = np.cos(x1/2)
    p2 = np.cos(x2/2)
    val = 2 + p1*p2
    return val**5


def chi2(x1, x2, obsexp, obssig):
    obsth = obs1(x1, x2)
    return ((obsth - obsexp)/obssig)**2


def lh(x1, x2, obsexp, obssig):
    exponent = -chi2(x1, x2, obsexp, obssig)/2
    return np.exp(exponent)


def chi2mlpred(mlpred, obsexp, obssig):
    return ((mlpred - obsexp)/obssig)**2


def suggestpts(npts, themodel, randpts=10, testini=10000):
    # npts: Number of points that will be suggested.
    # themodel: the model instance that was already trained.
    # randpts, optional: number of points chosen at random that will be added
    #   to the suggested points, randpts < npts
    # testini, optional: number of points that will be used to test the model
    #   in order to obtain suggested points, testini > npts

    # Try to  predict the observable for several points using what the machine
    # learned
    xtry = np.random.rand(testini, 2)*10*np.pi
    ptry = themodel.predict(xtry).flatten()

    # Select some candidate points based on the prediction from the model
    fltrcand = (ptry < 100 + 2*10)*(ptry > 100 - 2*10)
    xcand = xtry[fltrcand]

    # Use the points according to the observable predicted by the model but
    # pass the correct observable. In this way, the points that the model got
    # wrong should be corrected
    xsel = xcand[:npts - randpts]
    obs1sel = obs1(xsel[:, 0], xsel[:, 1])
    check1 = (obs1sel < 100 + 2*10)*(obs1sel > 100 - 2*10)
    check2 = ((ptry[fltrcand][:npts - randpts] - obs1sel)
              / (ptry[fltrcand][:npts - randpts] + obs1sel))
    stats1 = np.sum(check1)/obs1sel.shape[0]
    stats2 = np.sum(check2 < 1.e-3)/obs1sel.shape[0]
    stats3 = np.sum(check1*(check2 < 1.e-3))/obs1sel.shape[0]

    # Append `randpts` more points chosen at random
    xnew = np.random.rand(randpts, 2)*10*np.pi
    xout = np.append(xsel, xnew, axis=0)
    obs1out = np.append(obs1sel, obs1(xnew[:, 0], xnew[:, 1]))
    return xout, obs1out, stats1, stats2, stats3


# %% Attempt to do something I do not know

model = Sequential()
model.add(Dense(100, input_shape=(2,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# inputs = Input(shape=(2,))
# x1 = Dense(100, activation='relu')(inputs)
# x2 = Dense(100, activation='relu')(x1)
# x3 = Dense(100, activation='relu')(x2)
# x4 = Dense(100, activation='relu')(x3)
# outputs = Dense(1)(x4)
# model = Model(inputs=inputs, outputs=outputs)

# Mentioned in the paper but it is apparently the default
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='mse')

# Get the first set of 100 points
xsmpl = np.random.rand(100, 2)*10*np.pi
x1smpl = xsmpl[:, 0]
x2smpl = xsmpl[:, 1]
# Get the "observable" for those 100 points
obs1smpl = obs1(x1smpl, x2smpl)

# xval = np.random.rand(20, 2)*10*np.pi
# x1val = xval[:, 0]
# x2val = xval[:, 1]
# obs1val = obs1(x1val, x2val)

model.fit(xsmpl, obs1smpl, epochs=1000, batch_size=100, verbose=0)

# %% It apparently went OK

# Total number of retrainings, accumulated points will be 100*total + 100
total = 20
stats = np.zeros((total, 3))
for k in range(total):
    # Use `suggestpts` to get a suggestion of 100 new points
    xnext, obs1next, stats[k, 0], stats[k, 1], stats[k, 2] = \
        suggestpts(100, model, randpts=10, testini=10000)
    print(
        "Run", "{}/{}".format(k+1, total),
        ": Points actually within 2*sigma:",
        "{:.0%}".format(stats[k, 0]))

    # Append the suggested points to the previous points
    xsmpl = np.append(xsmpl, xnext, axis=0)
    obs1smpl = np.append(obs1smpl, obs1next)

    # Train again with all the accumulated points
    model.fit(
        xsmpl, obs1smpl,
        epochs=1000, batch_size=200, verbose=0)
    print("Points accumulated:", xsmpl.shape[0])

# Save the accumulated points to file
datafile = "x1_x2_sample_" + str(total) + ".csv"
np.savetxt(datafile, xsmpl)
# Save some stats:
#   fraction of points actually within 2*sigma,
#   fraction of points with error of less than 1e-3,
#   fraction of points within 2*sigma and error less than 1e-3
# statsfile = "stats_sample_" + str(total) + ".csv"
# np.savetxt(statsfile, stats)

print("Accumulated points saved to", datafile) 


# %%

xsmpl = np.loadtxt(datafile)

tstpts = 200*200
x1rand = np.random.rand(tstpts)*10*np.pi
x2rand = np.random.rand(tstpts)*10*np.pi
obs1rand = obs1(x1rand, x2rand)
lhrand = lh(x1rand, x2rand, 100, 10)
chi2rand = chi2(x1rand, x2rand, 100, 10)
predrand = model.predict(np.array([x1rand, x2rand]).transpose())
chi2pred = chi2mlpred(predrand, 100, 10)

plt.figure(figsize=(5, 5))
plt.title("Real result for " + str(tstpts) + " test points")
plt.scatter(x1rand, x2rand, c=obs1rand, s=1)
plt.colorbar()

plt.figure(figsize=(5, 5))
plt.title("ML model result for " + str(tstpts) + " test points")
plt.scatter(x1rand, x2rand, c=predrand[:, 0], s=1)
plt.colorbar()

plt.figure(figsize=(5, 5))
plt.title("Real likelihood for " + str(tstpts) + " test points")
plt.scatter(x1rand, x2rand, c=np.exp(-0.5*chi2rand), s=1)
plt.colorbar()

plt.figure(figsize=(5, 5))
plt.title("ML likelihood for " + str(tstpts) + " test points")
plt.scatter(x1rand, x2rand, c=np.exp(-0.5*chi2pred), s=1)
plt.colorbar()

plt.figure(figsize=(5, 5))
plt.title("Accumulated points in this run: " + str(xsmpl.shape[0]))
plt.scatter(xsmpl[:, 0], xsmpl[:, 1], s=1)
plt.show()
