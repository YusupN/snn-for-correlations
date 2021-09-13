import os
import librosa
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import math
import nest
from GRF import GRF
from math import sqrt
from math import exp
from math import pi
#from sklearn.model_selection import StratifiedKFold, KFold
#import generate_correlated_spike_trains.py

#
#parameters
#

#change this string to destination of your dataset
path = '/mnt/c/Users/user name/free-spoken-digit-dataset-master/recordings/'

n = 24
input_rate = 15
one_vector_longtitude = 2000
condition_on_weights = 0.3
dt = 0.1
n_fields = 7
n_splits = 5
GRF_use = False
max_stdp_iterations = 20
syn_spec = {
	'tau_plus': 89.3780097134877,
	'weight': {
				'distribution': 'uniform',
				'low': 0.0,
				'high': 1.0
			},
	'alpha': 1.6407236352330075,
	'model': "stdp_nn_restr_synapse",
	'Wmax': 1.0,
	'lambda': 0.001,
	'mu_plus': 0.0,
	'mu_minus': 0.0,
}

neuron_parameters = {
	'C_m': 1.5374180586077273,
	'I_e': 0.0,
	'V_th': -54.0,
	'tau_syn_in': 5.0,
	'tau_syn_ex': 5.0,
	'tau_minus': 59.96278052520938,
	'E_L': -70.0,
	't_ref': 3.0,
	'tau_m': 10.0,
}
#
#parameters
#

def generate_sample_train(rate, one_vector_longtitude):
    spikes_number_in_sample_is_ok = False
    while spikes_number_in_sample_is_ok == False:
        dt = 0.1
        n_steps = int(one_vector_longtitude / dt)
        p = rate * dt * 1e-3

        # Generate an array of float values uniformly distributed in [0; 1].
        sample_train = np.random.random(size=n_steps)
        # Get an array of bits.
        sample_train = sample_train < p
        exact_sample_train = np.array(range(n_steps))[sample_train] * dt
        if True:
        #if 31<len(exact_sample_train)<40:
            spikes_number_in_sample_is_ok = True
    return exact_sample_train, sample_train

def generate_trains(sample_train, correlations, rate, one_vector_longtitude, n, start_time=0):
    dt = 0.1
    n_steps = int(one_vector_longtitude / dt)
    p = rate * dt * 1e-3
    # Generally, theta and fi can be made different
    # for different trains.
    # So, we define them as arrays.
    thetas = [
        p + math.sqrt(c) * (1 - p)
        for c in correlations
    ]
    phis = [
        p * (1 - math.sqrt(c))
        for c in correlations
    ]

    binary_trains = [
        np.logical_or(
            np.logical_and(
                sample_train,
                np.random.random(size=n_steps) < theta
            ),
            np.logical_and(
                np.logical_not(sample_train),
                np.random.random(size=n_steps) < phi
            )
        )
        for theta, phi in zip(thetas, phis)
    ]
    # Manually set the first timestep not to contain a spike,
    # because NEST does not support zero spike times.
    # Hope it does not disturb the correlation much.
    for train in binary_trains:
        train[0] = False

    exact_spike_trains = [
        np.array(range(n_steps))[binary_train] * dt + start_time
        for binary_train in binary_trains
    ]
    exact_spike_trains = exact_spike_trains * n
    return exact_spike_trains


def get_data():
    data = load_iris()
    X = data.data
    Y = data.target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, stratify=Y)
    x_train = minmax_scale(x_train, feature_range = (0, 1), copy=False)
    x_test = minmax_scale(x_test, feature_range = (0, 1), copy=False)
    return x_train, x_test, y_train, y_test

def run_simul_of_one_class_neuron(stage, sample_train, input_vector, stdp_on, weights = None, condition_on_weights = condition_on_weights):
    number_of_inputs = n*len(input_vector[0])
    generators_ids = nest.Create('spike_generator', number_of_inputs)
    neuron = nest.Create('iaf_psc_exp', params=neuron_parameters)
    detector = nest.Create('spike_detector')
    input_detectors = nest.Create('spike_detector', number_of_inputs)
    inputs = nest.Create('parrot_neuron', number_of_inputs)
    nest.Connect(generators_ids, inputs, conn_spec = 'one_to_one', syn_spec = "static_synapse")
    nest.Connect(generators_ids, input_detectors, conn_spec = 'one_to_one', syn_spec = "static_synapse")
    if stdp_on == True:
        nest.Connect(inputs, neuron, conn_spec = 'all_to_all', syn_spec = syn_spec)
    if stdp_on == False:
        for k, weight in enumerate(weights):
            nest.Connect([inputs[k]], neuron, conn_spec = 'all_to_all', syn_spec = {
                'weight': weight,
                'model': "static_synapse",
                }
                )
    nest.Connect(neuron, detector, conn_spec = 'all_to_all', syn_spec = "static_synapse")
    total_time_elapsed = 0
    iteration_stdp = 0
    spike_trains = []
    wheights_are_not_settled = True
    while wheights_are_not_settled:
        for current_input_correlation_vector in input_vector:
            input_trains = generate_trains(
                sample_train,
                current_input_correlation_vector,
                input_rate,
                one_vector_longtitude,
                n,
                start_time=total_time_elapsed
            )
            nest.SetStatus(
                generators_ids,
                [
                    {'spike_times': input_train
                    }
                    for input_train in input_trains
                ]
            )
            nest.Simulate(one_vector_longtitude)
            total_time_elapsed = total_time_elapsed + one_vector_longtitude
            iteration_stdp = iteration_stdp + 1
            print(stage + ' step progress: ', iteration_stdp/len(input_vector))
            dSD = nest.GetStatus(detector, keys="events")[0]
            spike_times = dSD["times"]
            spike_times = [(st - total_time_elapsed + one_vector_longtitude) for st in spike_times]
            spike_trains = spike_trains + [spike_times]
            nest.SetStatus(detector, {'n_events': 0})
        weights = nest.GetStatus(nest.GetConnections(inputs, neuron), 'weight')
        wheights_are_not_settled = (min([abs(weight - 0.5) for weight in weights]) < condition_on_weights)
        if (iteration_stdp/len(input_vector) >= max_stdp_iterations) or stdp_on == False:
            wheights_are_not_settled = False
    nest.ResetKernel()
    if stdp_on == True:
        return weights, iteration_stdp/len(input_vector)
    if stdp_on == False:
        return spike_trains

def find_average_interval(x_spikes):
    intervals = [y - x for x,y in zip(x_spikes,x_spikes[1:])]
    average_interval = sum(intervals)/len(intervals)
    return [average_interval]

def one_class_neuron_cross_correlation(neurons_class, exact_sample_train, sample_train, stage, x_train, x_test, y_train, y_test):
    X_neurons_class_train = []
    #X_neurons_class_train = X_neurons_class_train[y_train == neurons_class]
    for x, y in zip(x_train, y_train):
        if y == neurons_class: X_neurons_class_train = X_neurons_class_train + [x]
    stage1 = '\nrun of run_simul_of_one_class_neuron for weights (step 1/3)'
    weights, stdp_attempts = run_simul_of_one_class_neuron(stage + stage1, sample_train, X_neurons_class_train, stdp_on = True, weights = None, condition_on_weights = condition_on_weights)
    stage1 = '\nrun of run_simul_of_one_class_neuron for x_train (step 2/3)'
    x_train = run_simul_of_one_class_neuron(stage + stage1, sample_train, x_train, stdp_on = False, weights = weights)
    stage1 = '\nrun of run_simul_of_one_class_neuron for x_test (step 3/3)'
    x_test = run_simul_of_one_class_neuron(stage + stage1, sample_train, x_test, stdp_on = False, weights = weights)
    result_train = []
    lens_train = []
    lens_test = []
    for x_spikes in x_train:
        result = 0
        lens_train = lens_train + [len(x_spikes)]
        for x_spike in x_spikes:
            for sample_spike in exact_sample_train:
                if (sample_spike<x_spike) and (x_spike<sample_spike +20):
                    result = result + 1
        result_train = result_train + [[result]]
    #[sum([((sample_spike<x_spike<sample_spike +20)) for sample_spike in exact_sample_train]) for x_spike in x_spikes]
    result_test = []
    for x_spikes in x_test:
        result = 0
        lens_test = lens_test + [len(x_spikes)]
        for x_spike in x_spikes:
            for sample_spike in exact_sample_train:
                if (sample_spike<x_spike) and (x_spike<sample_spike +20):
                    result = result + 1
        result_test = result_test + [[result]]
    
    x_train = result_train
    x_test = result_test
    return x_train, x_test, stdp_attempts, lens_train, lens_test, weights


def value_in_gauss_distrib_probability(x, average, stdev):
	exponent = exp(-((x-average)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def find_accuracy(KFoldIter, n_splits, exact_sample_train, sample_train, x_train, x_test, y_train, y_test):
    #exact_sample_train, sample_train = generate_sample_train(input_rate, one_vector_longtitude)
    number_of_classes = len(set(y_test))
    x_train_all = []
    x_test_all = []
    #x_train_all = [[]] * len(x_train)
    #x_test_all = [[]] * len(x_test)
    average_correlations = []
    std_correlations = []
    stdp_attemptss = []
    accuracy_before_test_estimation_paramaters_based_distributions_similarity = []
    accuracy_before_test_estimation_paramaters_based_on_max_value_of_corrs = []
    lens_trains = []
    lens_tests = []
    weightss = []
    for i in set(y_test):
        stage = 'KFold ' + str(KFoldIter) + ' out of ' + str(n_splits) + '\nrun of one_class_neuron_cross_correlation for class ' + str(i)
        x_train_i, x_test_i, stdp_attempts, lens_train, lens_test, weights = one_class_neuron_cross_correlation(i, exact_sample_train, sample_train, stage, x_train, x_test, y_train, y_test)
        lens_trains = lens_trains + [lens_train]
        lens_tests = lens_tests + [lens_test]
        stdp_attemptss = stdp_attemptss + [stdp_attempts]
        weightss = weightss + [weights]
        x_train_i= np.array(x_train_i)
        x_test_i= np.array(x_test_i)
        average_correlations_for_neuron_i = []
        std_correlations_for_neuron_i = []
        for l in set(y_test):
            l_train = [y==l for y in y_train]
            average_correlations_for_neuron_i = average_correlations_for_neuron_i + [np.average(x_train_i[l_train])]
            std_correlations_for_neuron_i = std_correlations_for_neuron_i + [np.std(x_train_i[l_train])]
        average_corrs_difference_for_neur_i = [(acn_i - average_correlations_for_neuron_i[i]) for acn_i in average_correlations_for_neuron_i]
        std_corrs_sum_for_neur_i = [(std_corr_i + std_correlations_for_neuron_i[i]) for std_corr_i in std_correlations_for_neuron_i]
        corrs_closeness_to_neurs_class = [acd_i/stdc_i for acd_i, stdc_i in zip(average_correlations_for_neuron_i, std_correlations_for_neuron_i)]
        corrs_closeness_to_neurs_class.remove(min(corrs_closeness_to_neurs_class))
        accuracy_before_test_estimation_paramater_based_on_distributions_similarity = min(corrs_closeness_to_neurs_class)
        accuracy_before_test_estimation_paramater_based_on_max_value_of_corrs = max(average_correlations_for_neuron_i)
        accuracy_before_test_estimation_paramaters_based_distributions_similarity = accuracy_before_test_estimation_paramaters_based_distributions_similarity + [accuracy_before_test_estimation_paramater_based_on_distributions_similarity]
        accuracy_before_test_estimation_paramaters_based_on_max_value_of_corrs = accuracy_before_test_estimation_paramaters_based_on_max_value_of_corrs + [accuracy_before_test_estimation_paramater_based_on_max_value_of_corrs]
        average_correlations = average_correlations + [average_correlations_for_neuron_i]
        std_correlations = std_correlations + [std_correlations_for_neuron_i]
        x_train_all.append(x_train_i)
        x_test_all.append(x_test_i)

        '''
        for n, tr in enumerate(x_train_i):
            x_train_all[n] = x_train_all[n] + tr
        for n, te in enumerate(x_test_i):
            x_test_all[n].append(te)
            print(x_test_all, te)
        '''
    x_train_all = np.transpose(x_train_all)
    x_test_all = np.transpose(x_test_all)
    x_train = x_train_all[0]
    x_test = x_test_all[0]
    predicted_cor_diff = []
    predicted_cor_gauss = []
    for i_test in x_test:
        test_iter = 0
        probabilities = []
        corr_differences = []
        for i_corr, i_aver, i_std in zip(i_test, average_correlations, std_correlations):
            probability_linked_values = [value_in_gauss_distrib_probability(i_corr, ii_aver, ii_std) for ii_aver, ii_std in zip(i_aver, i_std)]
            i_corr_probability = probability_linked_values[test_iter]/sum(probability_linked_values)
            corr_difference = abs(i_corr - i_aver[test_iter])
            test_iter = test_iter + 1
            corr_differences = corr_differences + [corr_difference]
            probabilities = probabilities + [i_corr_probability]
        predicted_cor_diff = predicted_cor_diff + [corr_differences.index(min(corr_differences))]
        predicted_cor_gauss = predicted_cor_gauss + [probabilities.index(max(probabilities))]
    corr_rule_acc_scr = accuracy_score(y_test, predicted_cor_diff)
    corr_rule_f1 = f1_score(y_test, predicted_cor_diff, average='micro')
    corr_gauss_rule_acc_scr = accuracy_score(y_test, predicted_cor_gauss)
    corr_gauss_rule_f1 = f1_score(y_test, predicted_cor_gauss, average='micro')
    #print("confusion matrix based on correlation rule: ", confusion_matrix(y_test, predicted_cor_diff))
    #lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    lr_list = [0.075]
    for learning_rate in lr_list:
        gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate)
        gb_clf.fit(x_train, y_train)
        y_test_pred = gb_clf.predict(x_test)
        #print("Learning rate: ", learning_rate)
        #print("Accuracy score: {0:.3f}".format(gb_clf.score(x_test, y_test)))
        grad_boost_f1 = f1_score(y_test, y_test_pred, average='micro')
        grad_boost_acc_sc = gb_clf.score(x_test, y_test)
        #print("Confusion matrix", confusion_matrix(y_test_pred, y_test))
    return corr_rule_acc_scr, grad_boost_acc_sc, corr_gauss_rule_acc_scr, corr_rule_f1, grad_boost_f1, corr_gauss_rule_f1, stdp_attemptss, average_correlations, std_correlations, accuracy_before_test_estimation_paramaters_based_distributions_similarity, accuracy_before_test_estimation_paramaters_based_on_max_value_of_corrs, lens_trains, lens_tests, exact_sample_train, weightss



nest.SetKernelStatus({
            'resolution': 0.1,
            'local_num_threads': 1,
})

#digits = load_digits()
#X = digits.images.reshape(len(digits.images), -1)
#Y = digits.target
#X, Y = read_wav_from_path(path, s)
#data = load_iris()
#X = data.data
#Y = data.target
X, Y = load_breast_cancer(return_X_y=True)
skf = StratifiedKFold(n_splits)
corr_rule_results = []
grad_boost_results = []
corr_gauss_rule_acc_scores = []
corr_rule_results_f1 = []
grad_boost_results_f1 = []
corr_gauss_rule_f1s = []
KFoldIter = 0
stdp_attemptsss = []
average_correlationss = []
std_correlationss = []
acc_estimation_param_on_distrib_sim = []
acc_estim_param_on_max_corr_value = []
lens_trainss = []
lens_testss = []
exact_sample_trains = []
weightsss = []


for stat in range(1):
    for train, test in skf.split(X, Y):
        exact_sample_train, sample_train = generate_sample_train(input_rate, one_vector_longtitude)
        KFoldIter = KFoldIter + 1
        x_train, x_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        x_train = minmax_scale(x_train, feature_range = (0, 1), copy=False)
        x_test = minmax_scale(x_test, feature_range = (0, 1), copy=False)
        if GRF_use == True:
            encode_with_GRF = GRF(n_fields)
            encode_with_GRF.fit(x_train)
            x_train = encode_with_GRF.transform(x_train)
            x_test = encode_with_GRF.transform(x_test)
        corr_rule_acc_scr, grad_boost_acc_sc, corr_gauss_rule_acc_scr, corr_rule_f1, grad_boost_f1, corr_gauss_rule_f1, stdp_attemptss, average_correlations, std_correlations, accuracy_before_test_estimation_paramaters_based_distributions_similarity, accuracy_before_test_estimation_paramaters_based_on_max_value_of_corrs, lens_trains, lens_tests, exact_sample_train, weightss = find_accuracy(KFoldIter, n_splits, exact_sample_train, sample_train, x_train, x_test, y_train, y_test)
        corr_rule_results = corr_rule_results + [corr_rule_acc_scr]
        grad_boost_results = grad_boost_results + [grad_boost_acc_sc]
        corr_gauss_rule_acc_scores = corr_gauss_rule_acc_scores + [corr_gauss_rule_acc_scr]
        corr_rule_results_f1 = corr_rule_results_f1 + [corr_rule_f1]
        grad_boost_results_f1 = grad_boost_results_f1 + [grad_boost_f1]
        corr_gauss_rule_f1s = corr_gauss_rule_f1s + [corr_gauss_rule_f1]
        stdp_attemptsss = stdp_attemptsss + [stdp_attemptss]
        average_correlationss = average_correlationss + [average_correlations]
        std_correlationss = std_correlationss + [std_correlations]
        acc_estimation_param_on_distrib_sim = acc_estimation_param_on_distrib_sim + [accuracy_before_test_estimation_paramaters_based_distributions_similarity]
        acc_estim_param_on_max_corr_value = acc_estim_param_on_max_corr_value + [accuracy_before_test_estimation_paramaters_based_on_max_value_of_corrs]
        lens_trainss = lens_trainss + [lens_trains]
        lens_testss = lens_testss + [lens_tests]
        exact_sample_trains = exact_sample_trains + [exact_sample_train]
        weightsss = weightsss + [weightss]
#print(len(sample_train))
#print(train, test)

print(np.average(corr_rule_results))
print(np.std(corr_rule_results))
print(corr_rule_results)
print(np.average(grad_boost_results))
print(np.std(grad_boost_results))
print(grad_boost_results)
print(np.average(corr_gauss_rule_acc_scores))
print(np.std(corr_gauss_rule_acc_scores))
print(corr_gauss_rule_acc_scores)
print(np.average(corr_rule_results_f1))
print(np.std(corr_rule_results_f1))
print(np.average(grad_boost_results_f1))
print(np.std(grad_boost_results_f1))
print(np.average(corr_gauss_rule_f1s))
print(np.std(corr_gauss_rule_f1s))
