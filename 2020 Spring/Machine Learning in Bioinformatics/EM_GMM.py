from numpy import log as ln
import math

class Gaussian:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def pdf(self, x):
        sigma = math.sqrt(self.var)
        return 1/(sigma*math.sqrt(2*math.pi))*math.exp(-1/2*math.pow(((x-self.mu)/sigma),2))


def log_likelihood(data, gaussians, pi):
    likelihood = 0
    for point in data:
        #print(likelihood, pi[0], gaussians[0].pdf(point), pi[1], gaussians[1].pdf(point))
        likelihood += ln(pi[0] * gaussians[0].pdf(point) + pi[1] * gaussians[1].pdf(point))
    return likelihood

def EM(iterations=5):
    #data = [3,4,5,5,6,8,12,12,13]
    data = [1,2,4,7,8,10]
    pi = [0.5,0.5]
    gaussians = [Gaussian(1,1), Gaussian(10,1)]
    
    print("Initial likelihood:", log_likelihood(data, gaussians, pi))
    
    for j in range(iterations):
        print("----------", "Iteration:", j, "-----------")
        
        tau = [[],[]]
        
        # compute responsibilities tau
        for i, point in enumerate(data):
            g_0 = gaussians[0].pdf(point)
            g_1 = gaussians[1].pdf(point)
            responsibility_0 = pi[0]*g_0/(pi[0]*g_0 + pi[1]*g_1)
            responsibility_1 = pi[1]*g_1/(pi[0]*g_0 + pi[1]*g_1)
            tau[0].append(responsibility_0)
            tau[1].append(responsibility_1)
            print(i, "𝜏:", "Gaussian 0:", responsibility_0, "Gaussian 1:", responsibility_1)
        
        # re-estimate parameters
        mu = [0,0]
        var = [0,0]
        N = [sum(tau[0]),sum(tau[1])]
        for i, point in enumerate(data):
            mu[0] += tau[0][i] * point
            mu[1] += tau[1][i] * point
        
        mu[0] = mu[0] / N[0]
        mu[1] = mu[1] / N[1]
        
        for i, point in enumerate(data):
            var[0] += tau[0][i] * math.pow(point - mu[0], 2)
            var[1] += tau[1][i] * math.pow(point - mu[1], 2)

        var[0] = var[0] / N[0]
        var[1] = var[1] / N[1]

        # set results for next iteration
        gaussians[0].mu = mu[0]
        gaussians[1].mu = mu[1]
        gaussians[0].var = var[0]
        gaussians[1].var = var[1]
        pi[0] = N[0] / len(data)
        pi[1] = N[1] / len(data)

        print("𝜇_0:", mu[0], "𝜇_1:", mu[1])
        print("𝚺_0:", var[0], "𝚺_1:", var[1])
        print("𝜋_0:", pi[0], "𝜋_1:", pi[1]) 
        print("N_0:", N[0], "N_1:", N[1])
        print("Log-likelihood:", log_likelihood(data, gaussians, pi))

EM()
