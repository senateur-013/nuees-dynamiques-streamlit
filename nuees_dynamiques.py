import numpy as np

class NueesDynamiques:
    def __init__(self, K, ni, max_iter=50, tol=1e-3):
        self.K = K
        self.ni = ni
        self.max_iter = max_iter
        self.tol = tol

    def initialisation(self, X):
        n = X.shape[0]
        indices = np.random.permutation(n)
        self.E = []
        pos = 0
        for _ in range(self.K):
            self.E.append(X[indices[pos:pos+self.ni]])
            pos += self.ni

    def distance(self, x, Ei):
        return np.mean(np.linalg.norm(Ei - x, axis=1))

    def affectation(self, X):
        classes = [[] for _ in range(self.K)]
        for x in X:
            distances = [self.distance(x, Ei) for Ei in self.E]
            idx = np.argmin(distances)
            classes[idx].append(x)
        return classes

    def nouvelle_nuee(self, Ci):
        Ci = np.array(Ci)
        centre = Ci.mean(axis=0)
        distances = np.linalg.norm(Ci - centre, axis=1)
        idx = np.argsort(distances)
        return Ci[idx[:self.ni]]

    def fit(self, X):
        self.initialisation(X)

        for _ in range(self.max_iter):
            classes = self.affectation(X)
            nouvelle_E = []

            for i in range(self.K):
                if len(classes[i]) >= self.ni:
                    nouvelle_E.append(self.nouvelle_nuee(classes[i]))
                else:
                    nouvelle_E.append(self.E[i])

            variation = sum(np.linalg.norm(self.E[i] - nouvelle_E[i])
                            for i in range(self.K))

            self.E = nouvelle_E

            if variation < self.tol:
                break

        self.C = classes
        return classes

