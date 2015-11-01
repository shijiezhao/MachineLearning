import svm
import numpy as np
import cvxopt
import plotFigure



"""
Some functions
"""

X = np.array([[1,2],[2,2],[0,0],[-2,3]]).copy()
Y = np.array([-1.,-1.,1.,1.]).copy()

trainer = svm.SVMTrainer(svm.Kernel.linear(),c=0.2)
predictor = trainer.train(X,Y)
plotFigure.plot(predictor, X,Y, grid_size=100, filename='TEST.pdf')

print predictor.predict(np.array([0,1]))


