from environment import *
from agent import *
import numpy
import math
from pylab import *
import matplotlib.pyplot as plt

num_timesteps = 1000
num_runs = 100
num_methods = 5
time_to_print = [10*i for i in range(101)]
errors = numpy.zeros((num_methods, num_runs, len(time_to_print)))

##### Settings for Random Walk ##########
features1 = numpy.array([[1.,0.,0.,0.,0.], 
                        [0.,1.,0.,0.,0.],
                        [0.,0.,1.,0.,0.],
                        [0.,0.,0.,1.,0.],
                        [0.,0.,0.,0.,1.]])
                        
features2 = numpy.array([[0.0, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.0, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.0, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.0]])
                        
features3 = numpy.array([[1.0,              0.0,              0.],
                         [1./math.sqrt(2.), 1./math.sqrt(2.), 0.],
                         [1./math.sqrt(3.), 1./math.sqrt(3.), 1./math.sqrt(3.)],
                         [0.0,              1./math.sqrt(2.), 1./math.sqrt(2.)],
                         [0.0,              0.0,              1.]])
                        
exp_rewards_walk = numpy.array([0., 0., 0., 0.5, 0.])

transit_prob_walk = numpy.array([[0. ,0.5,0. ,0. ,0.], 
                            [0. ,0. ,0.5,0. ,0.],
                            [1. ,0.5,0. ,0.5,1.],
                            [0. ,0. ,0.5,0. ,0.],
                            [0. ,0. ,0. ,0.5,0.]])
                            
###### Settings for 13-state Boyan Chain#########
featuresBoyan = numpy.array([[0., 0., 0.,   1.],
                             [0., 0., 0.25, 0.75],
                             [0., 0., 0.5,  0.5],
                             [0., 0., 0.75, 0.25],
                             [0., 0.,   1.,   0.],
                             [0., 0.25, 0.75, 0.],
                             [0., 0.5,  0.5,  0.],
                             [0., 0.75, 0.25, 0.],
                             [0., 1.,   0.,   0.],
                             [0.25, 0.75, 0., 0.],
                             [0.5,  0.5,  0., 0.],
                             [0.75, 0.25, 0., 0.],
                             [1.,   0.,   0., 0.]])
exp_rewards_Boyan = numpy.array([-3.]*13)
exp_rewards_Boyan[0] = 0.
exp_rewards_Boyan[1] = -2.
transit_prob_Boyan = numpy.zeros((13, 13))
transit_prob_Boyan[12][0] = 1.
transit_prob_Boyan[0][1] = 1.
for i in range(2, 13):
    transit_prob_Boyan[i-1][i] = 0.5
    transit_prob_Boyan[i-2][i] = 0.5


# square root of the objective function: mean-square projected Bellman error
# measured in 2-norm
# exp_rewards: numpy.array(n); transit_prob: numpy.array((n, n))
# exp_rewards[i]: expected rewards from state i
# transit_prob[i][j]: probability of transiting from state j to state i
def RMSPBE(gamma, weights, features, exp_rewards, transit_prob):
    V = numpy.dot(features, weights)
    TV = exp_rewards + gamma * numpy.dot(transit_prob, V)
    # projected
    PTV = numpy.linalg.inv( numpy.dot(features.transpose(), features) )
    PTV = numpy.dot( numpy.dot(features , PTV) , features.transpose())
    PTV = numpy.dot(PTV, TV)
    error = V - PTV
    return math.sqrt(numpy.dot(error, error) / len(error))
    

def findParas(features, domain):
    if domain == "RandomWalk":
        env = RandomWalk()
        exp_rewards = exp_rewards_walk
        transit_prob = transit_prob_walk
    elif domain == "BoyanChain":
        env = BoyanChain()
        exp_rewards = exp_rewards_Boyan
        transit_prob = transit_prob_Boyan
    alg = TD(features)
    alpha_to_test = [0.002*pow(2.,i) for i in range(5)]
    eta_to_test = [0.125, 0.25, 0.5, 1., 2., 4.]
    num_etas = len(eta_to_test)
    errors_paras = numpy.zeros((num_methods+3*(num_etas-1), num_runs, len(alpha_to_test)))
    for iMethod in range(num_methods + 3*(num_etas-1)):
        for iAlpha in range(len(alpha_to_test)):
            for run in range(num_runs):
                env.reset()
                alg.reset()
                if (run+1)%10 == 0:
                    print "Method: ", iMethod, "  run: ", run
                for timestep in range(num_timesteps):
                    if env.isAtGoal():
                        env.reset()
                    cur_state = env.observeState()
                    action = env.randomPolicy()
                    reward = env.takeAction(action)
                    next_state = env.observeState()
                    if iMethod == 0: #TD
                        alg.TDupdate(cur_state, next_state, reward, alpha_to_test[iAlpha])
                    elif iMethod == 1: #avTD
                        alg.avTD(cur_state, next_state, reward, alpha_to_test[iAlpha])
                    elif iMethod>=2 and iMethod<2+num_etas:
                        alg.GTD(cur_state, next_state, reward, alpha_to_test[iAlpha], eta_to_test[(iMethod-2)%num_etas]*alpha_to_test[iAlpha])
                    elif iMethod>=2+num_etas and iMethod<2+2*num_etas:
                        alg.GTD2(cur_state, next_state, reward, alpha_to_test[iAlpha], eta_to_test[(iMethod-2)%num_etas]*alpha_to_test[iAlpha])
                    elif iMethod>=2+2*num_etas and iMethod<2+3*num_etas:
                        alg.TDC(cur_state, next_state, reward, alpha_to_test[iAlpha], eta_to_test[(iMethod-2)%num_etas]*alpha_to_test[iAlpha])
                                        
                    errors_paras[iMethod][run][iAlpha] += RMSPBE(alg.gamma, alg.weights, features, exp_rewards, transit_prob)
    errors_paras /= num_timesteps
    avgerror = numpy.mean(errors_paras, axis = 1)
    fig, ax = plt.subplots()
    
    ax.plot(alpha_to_test, avgerror[0], label = "TD", color='r')
    ax.plot(alpha_to_test, avgerror[1], label = "avTD", color='m')
    
    alg_label = ["GTD", "GTD2", "TDC"]
    color_label = ['k', 'g', 'b']
    marker_label = [".", "o", "+", "", "v", "*"]
    num_algs = len(alg_label)
    num_markers = len(marker_label)
    for i in range(num_algs * num_markers):
        ax.plot(alpha_to_test, avgerror[2+i], 
                label = alg_label[i/num_markers]+", "+ r'$\eta=$'+repr(eta_to_test[i%num_markers]), 
                color = color_label[i/num_markers],
                marker = marker_label[i%num_markers])
    
    ax.legend(loc=0)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('RMSPBE')
    show()
    

def plot(features, domain):
    if domain == "RandomWalk":
        env = RandomWalk()
        exp_rewards = exp_rewards_walk
        transit_prob = transit_prob_walk
    elif domain == "BoyanChain":
        env = BoyanChain()
        exp_rewards = exp_rewards_Boyan
        transit_prob = transit_prob_Boyan
    alg = TD(features)
    for iMethod in range(num_methods):            
        for run in range(num_runs):
            env.reset()
            alg.reset()
            i = 0
            if (run+1)%10 == 0:
                print "Method: ", iMethod, "  run: ", run
            for timestep in range(num_timesteps+1):
                if env.isAtGoal():
                    env.reset()
                cur_state = env.observeState()
                action = env.randomPolicy()
                reward = env.takeAction(action)
                next_state = env.observeState()
                if iMethod == 0: #TD
                    alg.TDupdate(cur_state, next_state, reward, 0.008)
                elif iMethod == 1: # avTD
                    alg.avTD(cur_state, next_state, reward, 0.015)
                elif iMethod == 2: # GTD
                    alg.GTD(cur_state, next_state, reward, 0.04, 2.*0.04)
                elif iMethod == 3: # GTD2
                    alg.GTD2(cur_state, next_state, reward, 0.04, 2.*0.04)
                elif iMethod == 4: # TDC
                    alg.TDC(cur_state, next_state, reward, 0.04, 1.*0.04)
                
                if timestep == time_to_print[i]:
                    errors[iMethod][run][i] = RMSPBE(alg.gamma, alg.weights, features, exp_rewards, transit_prob)
                    i += 1

    avgerror = numpy.mean(errors, axis = 1)
    fig, ax = plt.subplots()

    ax.plot(time_to_print, avgerror[0], label = "TD", color='r')
    ax.plot(time_to_print, avgerror[1], label = "avTD", color='m')
    ax.plot(time_to_print, avgerror[2], label = "GTD",color='k')
    ax.plot(time_to_print, avgerror[3], label = "GTD2", color='g')
    ax.plot(time_to_print, avgerror[4], label = "TDC", color='b')

    ax.legend(loc=0)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('RMSPBE')
    ax.set_title('Boyan Chain')
    show()
    
    
if __name__ == "__main__":
    #findParas(featuresBoyan, "BoyanChain")
    #plot(features3, "RandomWalk")
    plot(featuresBoyan, "BoyanChain")