import numpy as np
import time
import logging
import sys
import select
import exceptions
import threading
from random import randint,uniform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ctypes import cdll


class ParamsInvalidException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class TorqueReachException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class BipolarSigmoidFunction(object):
    alpha = 2.0
    
    def Function(self, x):
        return 2 / (1 + np.exp(-self.alpha * x)) - 1
    
    def Derivative(self, x):
        y = self.Function(x)
        return self.alpha * (1 - y * y) / 2
    
    def Derivative2(self, y):
        return self.alpha * (1 - y * y) / 2

    def __init__(self, alpha):
        self.alpha = alpha

class Neuron(object):
    inputsCount = 0
    weights = None
    output = 0

    def __init__(self, inputs=0):
        self.inputsCount = max(1, inputs)
        self.weights = [0.0 for i in xrange(self.inputsCount)]
        self.Randomize()
        
    def Randomize(self):
        for i in xrange(self.inputsCount):
            self.weights[i] = np.random.random()

    def __getitem__(self, index):
        return self.weights[index]

    def __setitem__(self, index, value):
        self.weights[index] = value

    @property
    def Output(self):
        return self.output

    @property
    def InputsCount(self):
        return self.inputsCount

    def save(self, file_handle):
        for counter, weight in enumerate(self.weights):
            file_handle.write("Weight:%d:%f\n"%(counter, weight))

    def load(self, weights):
        print "I should not be called"
        for counter, weight in enumerate(self.weights):
            self.weights[counter] = weights[0][counter]
        
class Layer(object):
    inputsCount = 0
    neuronsCount = 0
    neurons = None
    output = None

    def __init__(self, neuronsCount=0, inputsCount=0):
        self.inputsCount = max(1, inputsCount)
        self.neuronsCount = max(1, neuronsCount)
        self.neurons = [Neuron() for i in xrange(self.neuronsCount)]
        self.output = [0.0 for i in xrange(self.neuronsCount)]

    def Compute(self, input):
        for i in xrange(self.neuronsCount):
            self.output[i] = self.neurons[i].Compute(input)
        return self.output

    def Randomize(self):
        for neuron in self.neurons:
            neuron.Randomize()

    @property
    def InputsCount(self):
        return self.inputsCount

    @property
    def NeuronsCount(self):
        return self.neuronsCount

    @property
    def Output(self):
        return self.output

    def __getitem__(self, index):
        return self.neurons[index]

    def save(self, file_handle):
        for counter, neuron in enumerate(self.neurons):
            file_handle.write("Neuron:%d\n"%counter)
            neuron.save(file_handle)

    def load(self, neurons):
        for counter, neuron in enumerate(self.neurons):
            neuron.load(neurons[counter])

class Network(object):
    inputsCount = 0
    layersCount = 0
    layers = None
    output = None

    def __init__(self, inputsCount, layersCount):
        self.inputsCount = max(1, inputsCount)
        self.layersCount = max(1, layersCount)
        self.layers = [Layer() for i in xrange(self.layersCount)]

    def Compute(self, input):
        self.output = input
        for layer in self.layers:
            self.output = layer.Compute(self.output)
        return self.output

    def Randomize(self):
        for layer in self.layers:
            layer.Randomize()
            
    @property
    def InputsCount(self):
        return self.inputsCount

    @property
    def LayersCount(self):
        return self.layersCount

    @property
    def Output(self):
        return self.output

    def __getitem__(self, index):
        return self.layers[index]

    def save(self, file_handle):
        for counter, layer in enumerate(self.layers):
            file_handle.write("Layer:%d\n"%counter)
            layer.save(file_handle)
        file_handle.write("Network Save Finished\n")


    def load(self, layers):
        for counter, layer in enumerate(self.layers):
            layer.load(layers[counter])
            

class ActivationNeuron(Neuron):
    threshold = 0.0
    function = None

    def __init__(self, inputs, function):
        super(ActivationNeuron, self).__init__(inputs)
        self.function = function

    def Randomize(self):
        super(ActivationNeuron, self).Randomize()
        self.threshold = np.random.random()

    def Compute(self, inputs):
        try:
            if len(inputs) != self.inputsCount:
                raise ValueError('argument validation failure')
            sum = 0.0
            for i in xrange(self.inputsCount):
                sum += self.weights[i] * inputs[i]
            sum += self.threshold

            self.output = self.function.Function(sum)
            return self.output
        except Exception,e:
            logging.exception(e)
            print e
            return None

    @property
    def ActivationFunction(self):
        return self.function

    @property
    def Threshold(self):
        return self.threshold

    @Threshold.setter
    def Threshold(self, value):
        self.threshold = value

    def save(self, file_handle):
        for counter, weight in enumerate(self.weights):
            file_handle.write("Weight:%d:%f\n"%(counter, weight))
        file_handle.write("Threshold:%f\n"%self.threshold)

    def load(self, weights):
        for counter, weight in enumerate(self.weights):
            self.weights[counter] = weights[0][counter]
        self.threshold = weights[1] 


class ActivationLayer(Layer):
    def __init__(self, neuronsCount, inputsCount, function):
        super(ActivationLayer, self).__init__(neuronsCount, inputsCount)
        for i in xrange(neuronsCount):
            self.neurons[i] = ActivationNeuron(inputsCount, function)
            

class ActivationNetwork(Network):
    minimal = 0.0
    maximal = 0.0
    srcfactor = 0.0
    dstfactor = 0.0
    def __init__(self, function, inputsCount, neuronsCountList):
        super(ActivationNetwork, self).__init__(inputsCount, len(neuronsCountList))
        for i in xrange(self.layersCount):
            self.layers[i] = ActivationLayer(neuronsCountList[i], 
                            inputsCount if i == 0 else neuronsCountList[i - 1], 
                            function)


class L2BackPropagationLearning(object):
    network = None
    learningRate = 0.1
    momentum = 0.0

    neuronErrors = None
    weightsUpdates = None
    thresholdsUpdates = None
    
    def __init__(self, l2_network):
        self.network = l2_network
        self.neuronErrors = [[] for i in xrange(l2_network.LayersCount)]
        self.weightsUpdates = [[] for i in xrange(l2_network.LayersCount)]
        self.thresholdsUpdates = [[] for i in xrange(l2_network.LayersCount)]

        self.l2_tolerence = 0.0

        for i in xrange(l2_network.LayersCount):
            layer = l2_network[i]
            self.neuronErrors[i] = [0.0 for j in xrange(layer.NeuronsCount)]
            self.weightsUpdates[i] = [[] for j in xrange(layer.NeuronsCount)]
            self.thresholdsUpdates[i] = [0.0 for j in xrange(layer.NeuronsCount)]

            for n in xrange(layer.NeuronsCount):
                self.weightsUpdates[i][n] = [0.0 for j in xrange(layer.InputsCount)]

    def Run(self, input, output):
        self.network.Compute(input)
        error = self.CalculateError(output)
        self.CalculateUpdates(input)
        self.UpdateNetwork()
        return error

    # it is a specific handover function
    def RunEpoch(self, input, output):
        error = 0.0
        for i in xrange(len(input)):
            error += self.Run(input[i], output[i])
        return error


    def CalculateError(self, desiredOutput):
        layersCount = self.network.LayersCount
        function = self.network[0][0].ActivationFunction
        error = 0
        
        layer = self.network[layersCount - 1]
        errors = self.neuronErrors[layersCount - 1]

        # last layer must be only 1 output for fan control purpose
        for i in xrange(layer.NeuronsCount):
            output = layer[i].Output
            e = desiredOutput[i] - output
            errors[i] = e * function.Derivative2(output)
            error += (e * e)

        j = layersCount - 2
        while j >= 0:
            layer = self.network[j]
            layerNext = self.network[j + 1]
            errors = self.neuronErrors[j]
            errorsNext = self.neuronErrors[j + 1]

            for i in xrange(layer.NeuronsCount):
                sum = 0.0
                for k in xrange(layerNext.NeuronsCount):
                    sum += errorsNext[k] * layerNext[k][i]
                errors[i] = sum * function.Derivative2( layer[i].Output )
            j -= 1
        
        return error / 2.0


    def CalculateUpdates(self, input):
        layer = self.network[0]
        errors = self.neuronErrors[0]
        layerWeightsUpdates = self.weightsUpdates[0]
        layerThresholdUpdates = self.thresholdsUpdates[0]

        for i in xrange(layer.NeuronsCount):
            neuron  = layer[i]
            error   = errors[i]
            neuronWeightUpdates = layerWeightsUpdates[i]

            for j in xrange(neuron.InputsCount):
                neuronWeightUpdates[j] = self.learningRate * (
                        self.momentum * neuronWeightUpdates[j] +
                        ( 1.0 - self.momentum ) * error * input[j]
            )

            layerThresholdUpdates[i] = self.learningRate * (
                self.momentum * layerThresholdUpdates[i] +
                ( 1.0 - self.momentum ) * error
                )

        for k in xrange(1, self.network.LayersCount):
            layerPrev = self.network[k - 1]
            layer = self.network[k]
            errors = self.neuronErrors[k]
            layerWeightsUpdates = self.weightsUpdates[k]
            layerThresholdUpdates = self.thresholdsUpdates[k]

            for i in xrange(layer.NeuronsCount):
                neuron  = layer[i]
                error   = errors[i]
                neuronWeightUpdates = layerWeightsUpdates[i]

                for j in xrange(neuron.InputsCount):
                    neuronWeightUpdates[j] = self.learningRate * (
                    self.momentum * neuronWeightUpdates[j] +
                    ( 1.0 - self.momentum ) * error * layerPrev[j].Output
                    )

                layerThresholdUpdates[i] = self.learningRate * (
                    self.momentum * layerThresholdUpdates[i] +
                    ( 1.0 - self.momentum ) * error
                )

    def UpdateNetwork(self):
        for i in xrange(self.network.LayersCount):
            layer = self.network[i]
            layerWeightsUpdates = self.weightsUpdates[i]
            layerThresholdUpdates = self.thresholdsUpdates[i]

            for j in xrange(layer.NeuronsCount):
                neuron = layer[j]
                neuronWeightUpdates = layerWeightsUpdates[j]

                for k in xrange(neuron.InputsCount):
                    neuron[k] += neuronWeightUpdates[k]
                neuron.Threshold += layerThresholdUpdates[j]

                
def InitDevControlNode():
    global simulate
    if simulate:
        return None
    epos = cdll.LoadLibrary('./libepos2.so');
    # 1: current mode, 0: velocity mode
    if epos.DeviceInitialize(1) == 0:
        return epos
    else:
        print "Open Device Control file failure"
        return None
#utils    


simulate_velocity = 0
simulate_torque = 0

def PreciseDutyCycle(dutycycle):
    return round(dutycycle, 3)

def RangeDutyCycle():
    dutycyclerange = []
    for i in xrange(1001):
        dutycyclerange.append(float(i) / 1000)
    return dutycyclerange

def ConfigDutyCycle(dutycycle, brake=False):
    global screw_dev
    global simulate
    global simulate_velocity

    BASE_FACTOR = 0
    MAX_FACTOR = 1000
    rtval = False
    #screw machine is controlled by velocity, the range of velocity is from 0 to 1000
    velocity_range = MAX_FACTOR - BASE_FACTOR
    velocity = velocity_range * dutycycle + BASE_FACTOR
    if simulate:
        simulate_velocity = int(velocity)
    elif screw_dev.DeviceSetTargetCurrent(int(velocity)) == 0:
        rtval = True

#    print "set target current %d"%(int(velocity))
        
    graphLock.acquire()
    velocityQueue.append(velocity)
    if len(velocityQueue) > VELOCITY_X_RANGE:
        velocityQueue.pop(0)
    graphLock.release()

    return rtval
    

from datetime import datetime
start_time = 0

def millis():
    global start_time
    dt = datetime.now() - start_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms


#measured current :     510, velocity = 460
#measured current :     710, velocity = 900
#measured current :     600, velocity = 120
def TorqueFormular(current, velocity):
    global L1_EXPECT_MAXIMAL_TORQUE
    TORQUE_FACTOR = 100.0
    torque = L1_EXPECT_MAXIMAL_TORQUE
    if velocity > 0:
        torque = float(float(current) * TORQUE_FACTOR) / float(velocity)
        if torque > L1_EXPECT_MAXIMAL_TORQUE:
            torque = L1_EXPECT_MAXIMAL_TORQUE
#    print "current:%d, velocity:%d, torque:%.5f"%(current, velocity, torque)
    return torque

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def ReadTorque():
    global start_time
    global L1_EXPECT_MAXIMAL_TORQUE
    global L1_EXPECT_MINIMAL_TORQUE
    global screw_dev
    global simulate
    global simulate_velocity

    if simulate:
        torque_step = float((L1_EXPECT_MAXIMAL_TORQUE - L1_EXPECT_MINIMAL_TORQUE)) / 1000
        torque_range = list(frange(L1_EXPECT_MINIMAL_TORQUE, L1_EXPECT_MAXIMAL_TORQUE, torque_step))
        torque_range.reverse()
#        print torque_range
        torque = torque_range[simulate_velocity]
    else:
        if start_time == 0:
            start_time =  datetime.now()
        else:
            print "torque elaps:%d"%millis()
            start_time = datetime.now()
            
        torque = 0.0
        if screw_dev is not None:
            torque = screw_dev.DeviceGetCurrentNow()
    if torque < 0:
        torque = 0
            
    graphLock.acquire()
    graphQueue.append(torque)
    if len(graphQueue) > GRAPH_X_RANGE:
        graphQueue.pop(0)
    graphLock.release()

    return torque
    
def SaveNetwork(mynetwork, filename, minimal, maximal, srcfactor, dstfactor):
    local_file = open(filename, "w")
    if local_file is None: return

    local_file.write("MIN:%f\n"%minimal)
    local_file.write("MAX:%f\n"%maximal)
    local_file.write("SRCFACTOR:%f\n"%srcfactor)
    local_file.write("DSTFACTOR:%f\n"%dstfactor)
    
    mynetwork.save(local_file)
    local_file.close()

def RestoreNetwork(filename, inputCount):
    sigmoidAlphaValue = 2.0
    
    local_file = open(filename, "r")
    line = local_file.readline()

    minimal = 0.0
    maximal = 0.0
    srcfactor = 0.0
    dstfactor = 0.0
    #parse parameters firstly
    while "MIN:" in line or "MAX:" in line or "SRCFACTOR:" in line or "DSTFACTOR:" in line:
        if "MIN:" in line:
            tag,value = line.split(':')
            value = ''.join(value.splitlines())
            minimal = float(value)
        if "MAX:" in line:
            tag,value = line.split(':')
            value = ''.join(value.splitlines())
            maximal = float(value)
        if "SRCFACTOR:" in line:
            tag,value = line.split(':')
            value = ''.join(value.splitlines())
            srcfactor = float(value)
        if "DSTFACTOR:" in line:
            tag,value = line.split(':')
            value = ''.join(value.splitlines())
            dstfactor = float(value)
        line = local_file.readline()
        
    mynetwork = None

    if local_file is not None:
        layers = []
        while "Layer" in line:
            line = local_file.readline()
            neuron = []
            while "Neuron" in line:
                line = local_file.readline()
                weights = []
                threshold = "0.0"
                while "Weight" in line:
                    tag,count,value = line.split(':')
                    value = ''.join(value.splitlines())
                    weights.append(float(value))
                    line = local_file.readline()
                while "Threshold" in line:
                    tag,threshold = line.split(':')
                    threshold = ''.join(threshold.splitlines())
                    line = local_file.readline()
                neuron.append((weights, float(threshold)))
            layers.append(neuron)

        local_file.close()

        param_layers = []
        for layer in layers:
            param_layers.append(len(layer))

        print "construct network" + str(param_layers)

        mynetwork = ActivationNetwork(BipolarSigmoidFunction(sigmoidAlphaValue),\
                inputCount, param_layers)
        mynetwork.load(layers)

        # set parameters
        print "network parameters:\n minimal:%f, maximal:%f, srcfactor:%f, \
                dstfactor:%f"%(minimal, maximal, srcfactor, dstfactor)
                    
    return (mynetwork, minimal, maximal, srcfactor, dstfactor)

class Monitor(threading.Thread):
    quit = False
    def __init__(self, t_name):    
        threading.Thread.__init__(self, name = t_name)
    def run(self):
        while not self.quit:
            ReadTorque()
    def terminate(self):
        self.terminate = True

def LoadSamples(filename):
    params = {}
    with open(filename) as mapfile:
        for line in mapfile:
            a,b = line.strip().split(',')
            params[int(a)] = int(b)
    return params
    

def init():
    global line
    global line2
    
    line.set_ydata(graphQueue)
    line2.set_ydata(velocityQueue)
    return line,line2

def update(data):
    global graphQueue
    global velocityQueue
    global line
    global line2
    
    graphLock.acquire()
    line.set_ydata(graphQueue)
    line2.set_ydata(velocityQueue)
    graphLock.release()
    return line,line2

def data_gen():
    while True:
        yield graphQueue    

class ScrewDriver(threading.Thread):
    m_minx = 0
    m_maxx = 0
    m_miny = 0
    m_maxy = 0

    m_velminx = 0
    m_velmaxx = 100
    m_velminy = 0
    m_velmaxy = 700
    
    def __init__(self, t_name):    
        threading.Thread.__init__(self, name = t_name)
    def run(self):
        global GRAPH_X_RANGE
        global VELOCITY_X_RANGE
        global graphQueue
        global velocityQueue
        global graphCon
        global fig
        global axes1
        global screw_dev
        
        if len(parameters) > 1:
            if "create_network" in parameters[1]:
                sample_file = parameters[2]
                samples = None
                if sample_file is not None:
                    samples = LoadSamples(sample_file)
                else:
                    raise ParamsInvalidException('invalid sample file')
                if samples is None:
                    raise ParamsInvalidException('invalid sampless')
                # pure mathmatic calculation
                listx = [key for key,value in samples.iteritems()]
                listy = [value for key,value in samples.iteritems()]
                listx.sort()
                listy.sort()
                minx = listx[0]
                maxx = listx[-1]
                miny = listy[0]
                maxy = listy[-1]

                print listx, listy
                print "Press Enter when you are ready"
                raw_input()

                graphCon.acquire()
                #graphics support
                GRAPH_X_RANGE = 1000
                graphQueue = [miny for i in xrange(maxx - minx)]
                VELOCITY_X_RANGE = 1000
                velocityQueue = [0 for i in xrange(VELOCITY_X_RANGE)]
                self.m_minx = minx
                self.m_maxx = maxx
                self.m_miny = miny
                self.m_maxy = maxy
                graphCon.notify()
                graphCon.release()

                
                sigmoidAlphaValue = 2.0
                l2_network = ActivationNetwork(BipolarSigmoidFunction(sigmoidAlphaValue), 1, [20, 1])
                l2_teacher = L2BackPropagationLearning(l2_network)
                print "Please input running iterations:\n"
                series = int(''.join(raw_input().splitlines()))
                print "interations:%d"%series

                dstfactor = 1.7 / (maxy - miny)
                srcfactor = 2.0 / ((maxx - minx) + 1)

                inparameters = [[(i - minx) * srcfactor - 1.0] for i in listx]
                outparameters = [[(samples[i] - miny) * dstfactor - 0.85] for i in listx]
                
                while series > 0:
                    #run 1 serial
                    print "This interation:%d"%series
                    series -= 1

                    err = l2_teacher.RunEpoch(inparameters, outparameters)
                    print err

                    if (series % 10) == 0:                        
                        graphLock.acquire()
                        graphQueue = [(l2_network.Compute([i * srcfactor - 1.0])[0] + 0.85) / dstfactor + miny for i in xrange(maxx - minx)]
                        graphLock.release()

                l2filename = sample_file + '_network'
                SaveNetwork(l2_network, l2filename, minx, maxx, srcfactor, dstfactor);

                curvfile = open('curvfile.txt', 'w');
                for m in xrange(maxx - minx):
                    n = (l2_network.Compute([m * srcfactor - 1.0])[0] + 0.85) / dstfactor + miny
                    curvfile.write("%d," % n)
                curvfile.close()					
                
            elif "screwdriver" in parameters[1]:
                target_torque = parameters[2]
                max_speed = parameters[3]
                max_speed = int(max_speed)
                maplist = [993,992,992,992,991,991,990,990,990,989,989,988,988,988,987,987,986,986,986,985,985,984,984,984,983,983,982,982,981,981,981,980,980,979,979,978,978,978,977,977,976,976,975,975,975,974,974,973,973,972,972,972,971,971,970,970,969,969,968,968,967,967,967,966,966,965,965,964,964,963,963,962,962,962,961,961,960,960,959,959,958,958,957,957,956,956,955,955,954,954,954,953,953,952,952,951,951,950,950,949,949,948,948,947,947,946,946,945,945,944,944,943,943,942,942,941,941,940,940,939,939,938,938,937,937,936,936,936,935,935,934,934,933,933,932,932,931,931,930,930,929,929,928,928,927,927,926,926,925,925,924,924,923,923,922,922,921,920,920,919,919,918,918,917,917,916,916,915,915,914,914,913,913,912,912,911,911,910,910,909,909,908,908,907,907,906,906,905,905,904,904,903,903,902,902,901,901,900,900,899,899,898,898,897,897,896,896,895,895,894,894,893,893,892,892,891,891,890,890,889,889,888,888,887,887,886,886,886,885,885,884,884,883,883,882,882,881,881,880,880,879,879,878,878,877,877,876,876,875,875,874,874,873,873,872,872,872,871,871,870,870,869,869,868,868,867,867,866,866,865,865,864,864,864,863,863,862,862,861,861,860,860,859,859,858,858,858,857,857,856,856,855,855,854,854,854,853,853,852,852,851,851,850,850,849,849,849,848,848,847,847,846,846,846,845,845,844,844,843,843,842,842,842,841,841,840,840,839,839,839,838,838,837,837,836,836,836,835,835,834,834,833,833,833,832,832,831,831,830,830,830,829,829,828,828,827,827,827,826,826,825,825,825,824,824,823,823,822,822,822,821,821,820,820,819,819,819,818,818,817,817,816,816,816,815,815,814,814,814,813,813,812,812,811,811,811,810,810,809,809,808,808,808,807,807,806,806,805,805,804,804,804,803,803,802,802,801,801,800,800,800,799,799,798,798,797,797,796,796,795,795,795,794,794,793,793,792,792,791,791,790,790,789,789,788,788,787,787,786,786,786,785,785,784,784,783,783,782,782,781,780,780,779,779,778,778,777,777,776,776,775,775,774,774,773,773,772,771,771,770,770,769,769,768,767,767,766,766,765,765,764,763,763,762,762,761,760,760,759,758,758,757,757,756,755,755,754,753,753,752,751,751,750,749,749,748,747,747,746,745,744,744,743,742,742,741,740,739,739,738,737,736,736,735,734,733,733,732,731,730,729,729,728,727,726,725,725,724,723,722,721,720,719,719,718,717,716,715,714,713,712,711,711,710,709,708,707,706,705,704,703,702,701,700,699,698,697,696,695,694,693,692,691,690,689,688,687,686,685,683,682,681,680,679,678,677,676,674,673,672,671,670,669,667,666,665,664,663,661,660,659,658,656,655,654,653,651,650,649,647,646,645,643,642,641,639,638,637,635,634,632,631,630,628,627,625,624,622,621,620,618,617,615,614,612,611,609,607,606,604,603,601,600,598,596,595,593,592,590,588,587,585,583,582,580,578,577,575,573,572,570,568,566,565,563,561,559,558,556,554,552,550,549,547,545,543,541,539,537,536,534,532,530,528,526,524,522,520,518,516,514,512,510,509,507,505,503,501,498,496,494,492,490,488,486,484,482,480,478,476,474,472,470,467,465,463,461,459,457,455,453,450,448,446,444,442,440,437,435,433,431,429,426,424,422,420,418,415,413,411,409,406,404,402,400,397,395,393,391,388,386,384,382,379,377,375,373,370,368,366,363,361,359,357,354,352,350,348,345,343,341,338,336,334,332,329,327,325,323,320,318,316,313,311,309,307,304,302,300,298,295,293,291,289,286,284,282,280,278,275,273,271,269,267,264,262,260,258,256,253,251,249,247,245,243,241,238,236,234,232,230,228,226,224,222,220,217,215,213,211,209,207,205,203,201,199,197,195,193,191,189,187,185,183,181,180,178,176,174,172,170,168,166,164,163,161,159,157,155,154,152,150,148,146,145,143,141,139,138,136,134,133,131,129,128,126,124,123,121,119,118,116,115,113,112,110,108,107,105,104,102,101,99,98,96,95,94,92,91,89,88,86,85,84,82,81,80,78,77,76,74,73,72,70,69,68,67,65,64,63,62,60,59,58,57,56,54,53,52,51,50,49,48,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,22,21,20,19,18,17,16,15,15,14,13,12,11,10,10,9,8,7,7,6,5,4,4,3,2,1,1,0,0,0,0,0]
                torqueMapList = [0,0,0,0,0,0,1,1,1,2,2,3,3,3,4,4,4,5,5,6,6,6,7,7,7,8,8,9,9,9,10,10,10,11,11,12,12,12,13,13,14,14,14,15,15,16,16,17,17,17,18,18,19,19,19,20,20,21,21,22,22,22,23,23,24,24,25,25,26,26,26,27,27,28,28,29,29,30,30,30,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,38,39,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,49,49,50,50,51,51,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,60,60,61,62,62,63,63,64,64,65,65,66,66,67,68,68,69,69,70,70,71,72,72,73,73,74,74,75,76,76,77,77,78,78,79,80,80,81,81,82,83,83,84,84,85,85,86,87,87,88,88,89,90,90,91,91,92,93,93,94,95,95,96,96,97,98,98,99,99,100,101,101,102,103,103,104,104,105,106,106,107,108,108,109,110,110,111,111,112,113,113,114,115,115,116,117,117,118,119,119,120,120,121,122,122,123,124,124,125,126,126,127,128,128,129,130,130,131,132,132,133,134,134,135,136,136,137,138,138,139,140,140,141,142,142,143,144,144,145,146,146,147,148,148,149,150,150,151,152,153,153,154,155,155,156,157,157,158,159,159,160,161,161,162,163,163,164,165,166,166,167,168,168,169,170,170,171,172,172,173,174,174,175,176,177,177,178,179,179,180,181,181,182,183,183,184,185,186,186,187,188,188,189,190,190,191,192,192,193,194,195,195,196,197,197,198,199,199,200,201,201,202,203,204,204,205,206,206,207,208,208,209,210,210,211,212,213,213,214,215,215,216,217,217,218,219,219,220,221,221,222,223,224,224,225,226,226,227,228,228,229,230,230,231,232,232,233,234,234,235,236,236,237,238,238,239,240,240,241,242,242,243,244,244,245,246,246,247,248,248,249,250,250,251,252,252,253,254,254,255,256,256,257,258,258,259,260,260,261,262,262,263,264,264,265,265,266,267,267,268,269,269,270,271,271,272,273,273,274,274,275,276,276,277,278,278,279,280,280,281,281,282,283,283,284,285,285,286,286,287,288,288,289,289,290,291,291,292,293,293,294,294,295,296,296,297,297,298,299,299,300,300,301,302,302,303,303,304,305,305,306,306,307,308,308,309,309,310,311,311,312,312,313,313,314,315,315,316,316,317,317,318,319,319,320,320,321,321,322,323,323,324,324,325,325,326,327,327,328,328,329,329,330,330,331,332,332,333,333,334,334,335,335,336,337,337,338,338,339,339,340,340,341,341,342,342,343,344,344,345,345,346,346,347,347,348,348,349,349,350,350,351,351,352,353,353,354,354,355,355,356,356,357,357,358,358,359,359,360,360,361,361,362,362,363,363,364,364,365,365,366,366,367,367,368,368,369,369,370,370,371,371,372,372,373,373,374,374,375,375,376,376,377,377,378,378,379,379,380,380,381,381,382,382,383,383,384,384,385,385,386,386,387,387,388,388,389,389,390,390,391,391,392,392,392,393,393,394,394,395,395,396,396,397,397,398,398,399,399,400,400,401,401,402,402,403,403,403,404,404,405,405,406,406,407,407,408,408,409,409,410,410,411,411,412,412,412,413,413,414,414,415,415,416,416,417,417,418,418,419,419,420,420,421,421,421,422,422,423,423,424,424,425,425,426,426,427,427,428,428,429,429,430,430,431,431,431,432,432,433,433,434,434,435,435,436,436,437,437,438,438,439,439,440,440,441,441,442,442,443,443,443,444,444,445,445,446,446,447,447,448,448,449,449,450,450,451,451,452,452,453,453,454,454,455,455,456,456,457,457,458,458,459,459,460,460,461,461,462,462,463,463,464,464,465,465,466,466,467,467,468,468,469,469,470,470,471,471,472,472,473,473,474,474,475,475,476,476,477,477,478,478,479,479,480,481,481,482,482,483,483,484,484,485,485,486,486,487,487,488,488,489,489,490,491,491,492,492,493,493,494,494,495,495,496,496,497,498,498,499,499,500,500,501,501,502,502,503,504,504,505,505,506,506,507,507,508,509,509,510,510,511,511,512,513,513,514,514,515,515,516,516,517,518,518,519,519,520,520,521,522,522,523,523,524,525,525,526,526,527,527,528,529,529,530,530,531,532,532,533,533,534,534,535,536,536,537,537,538,539,539,540,540,541,542,542,543,543,544,545,545,546,546,547,548,548,549,549,550,551,551,552,553,553,554,554,555,556,556,557,557,558,559,559,560,561,561,562,562,563,564,564,565,566,566,567,567,568,569,569,570,571,571,572,572,573,574,574,575,576,576,577,578,578,579,579,580,581,581,582,583,583,584,585,585,586,587,587,588,589,589,590,590,591,592,592,593,594,594,595,596,596,597,598,598,599,600,600,601,602,602,603,604,604,605,605,606,607,607,608,609,609,610,611,611,612,613,613,614,615,615,616,617,617,618,619,619,620,621,621,622,623,623,624,625,625,626,627,627,628,629,629,630,631,631,632,633,633,634,635,635,636,637,637,638,639,639,640,641,641,642,643,643,644,645,645,646,647,648,648,649,650,650,651,652,652,653,654,654,655,656,656,657,658,658,659,660,660,661,662,662,663,664,664,665,666,666,667,668,668,669,670,670,671,672,672,673,674,674,675,676,676,677,678,678,679,680,680,681,682,682,683,684,684,685,686,686,687,688,688,689,690,690,691,692,692,693,694,694,695,696,696,697,698,698,699,700,700,701,702,702,703,704,704,705,705,706,707,707,708,709,709,710,711,711,712,713,713,714,715,715,716,717,717,718,718,719,720,720,721,722,722,723,724,724,725,725,726,727,727,728,729,729,730,731,731,732,732,733,734,734,735,736,736,737,737,738,739,739,740,741,741,742,742,743,744,744,745,746,746,747,747,748,749,749,750,750,751,752,752,753,753,754,755,755,756,756,757,758,758,759,759,760,761,761,762,762,763,764,764,765,765,766,767,767,768,768,769,769,770,771,771,772,772,773,774,774,775,775,776,776,777,778,778,779,779,780,780,781,782,782,783,783,784,784,785,785,786,787,787,788,788,789,789,790,790,791,791,792,793,793,794,794,795,795,796,796,797,797,798,798,799,800,800,801,801,802,802,803,803,804,804,805,805,806,806,807,807,808,808,809,809,810,810,811,811,812,812,813,813,814,814,815,815,816,816,817,817,818,818,819,819,820,820,821,821,822,822,823,823,824,824,825,825,826,826,827,827,828,828,829,829,829,830,830,831,831,832,832,833,833,834,834,835,835,835,836,836,837,837,838,838,839,839,840,840,840,841,841,842,842,843,843,843,844,844,845,845,846,846,846,847,847,848,848,849,849,849,850,850,851,851,852,852,852,853,853,854,854,854,855,855,856,856,856,857,857,858,858,858,859,859,860,860,860,861,861,862,862,862,863,863,863,864,864,865,865,865,866,866,866,867,867,868,868,868,869,869,869,870,870,871,871,871,872,872,872,873,873,873,874,874,874,875,875,875,876,876,877,877,877,878,878,878,879,879,879,880,880,880,881,881,881,882,882,882,883,883,883,884,884,884]
                target_torque = int(target_torque)
                target_current = torqueMapList[target_torque]
                
                print "Target Current:%d, Max Speed:%d"%(target_current, max_speed)

                graphCon.acquire()
                #graphics support
                GRAPH_X_RANGE = 2000
                graphQueue = [target_current for i in xrange(GRAPH_X_RANGE)]
                self.m_minx = 0
                self.m_maxx = GRAPH_X_RANGE
                self.m_miny = 0
                self.m_maxy = target_current + 100

                VELOCITY_X_RANGE = 2000
                velocityQueue = [0 for i in xrange(VELOCITY_X_RANGE)]
                self.m_velminx = 0
                self.m_velmaxx = VELOCITY_X_RANGE
                self.m_velminy = 0
                self.m_velmaxy = 10000
                graphCon.notify()
                graphCon.release()

#                screw_dev.DeviceSetCurrentPGain(1142)
                print "pGain:%d"%screw_dev.DeviceGetCurrentPGain()
                print "iGain:%d"%screw_dev.DeviceGetCurrentIGain()
 
                print "Press Enter when you are ready"
                raw_input()
                
                # apprach1 
                '''
                current_scale = 2
                base_target = target_current / current_scale
                left_target = target_current - base_target
                step = 1
                i = 0

                screw_dev.DeviceSetContinuousCurrentLimit(target_current)
                screw_dev.DeviceSetCurrentModeSpeedLimitation(max_speed)

                while i < base_target:
                    screw_dev.DeviceSetTargetCurrent(i) 
                    current = screw_dev.DeviceGetCurrentNow()
                    velocity = screw_dev.DeviceGetVelocityNow()
                    if velocity < 100 and i > 10:
                        step *= 2
                    else:
                        step = 1;
                    i += step
                    graphLock.acquire()
                    graphQueue.append(current)
                    if len(graphQueue) > GRAPH_X_RANGE:
                        graphQueue.pop(0)
                    velocityQueue.append(velocity)
                    if len(velocityQueue) > VELOCITY_X_RANGE:
                        velocityQueue.pop(0)
                    graphLock.release()
                

                stable = 0;
                stable_count = 1
                current_bias = 5
                overload = False
                movingCurrent = 99999
                stallCurrent = 0
                overloadTimes = 0
                oldSpeed = 0

                screw_dev.DeviceSetCurrentModeSpeedLimitation(max_speed)
#                screw_dev.DeviceSetMaxAcceleration(max_speed)
                while left_target > 20:
                    getout = False
                    screw_dev.DeviceSetTargetCurrent(base_target)
                    while not getout:
                        current = screw_dev.DeviceGetCurrentNow()
                        velocity = screw_dev.DeviceGetVelocityNow();
                        if abs(current - base_target) > 5:
                            stable = 0
                        else:
                            stable += 1

                        if stable > stable_count:
                            stable = 0
                            velocity = screw_dev.DeviceGetVelocityNow();
                            if velocity < 300:
                                getout = True
                                if velocity < 100:
                                    stallCurrent = current
                                    stalled = True
                                    print "Stalled at: %d, target: %d"%(current, base_target)
                                else:
                                    movingCurrent = current
                            else:
                                movingCurrent = current
                        
                        if stallCurrent > movingCurrent and \
                            abs(stallCurrent - movingCurrent) > 30:
                                overload = True
                                overloadTimes += 1
                                print "stallCurrent:%d, movingCurrent:%d"%(stallCurrent, movingCurrent)

                        if overload:
                            overload = False
                            movingCurrent = 99999
                            limit_speed = 1000/overloadTimes
                            if limit_speed < 500:
                                limit_speed = 500
                            if oldSpeed != limit_speed:
                                oldSpeed = limit_speed
                                print "limited speed at:%d"%limit_speed
                                accl = limit_speed/60
#                                screw_dev.DeviceSetMaxAcceleration(accl)
                                screw_dev.DeviceSetCurrentModeSpeedLimitation(limit_speed)

                        graphLock.acquire()
                        graphQueue.append(current)
                        if len(graphQueue) > GRAPH_X_RANGE:
                            graphQueue.pop(0)
                        velocityQueue.append(velocity)
                        if len(velocityQueue) > VELOCITY_X_RANGE:
                            velocityQueue.pop(0)
                        graphLock.release()
                    
                    base_target = base_target + 5
                    left_target = target_current - base_target

                print "Slowly Move to the Target"
                for i in xrange(left_target):
                    target = base_target + i
                    screw_dev.DeviceSetTargetCurrent(target)


                '''

                # aproach2
                out = False
                screw_dev.DeviceSetContinuousCurrentLimit(target_current)
                screw_dev.DeviceSetCurrentModeSpeedLimitation(1000)
                curMaxCurrent = 0
                wantCurrent = 0
                wantCurrent = target_current
                stableCount = 0
                stableDebounce = 3
                screw_dev.DeviceSetTargetCurrent(wantCurrent)
                while not out:
                    current_now = screw_dev.DeviceGetCurrentNow()
                    if current_now < 0: current_now = 0
                    if curMaxCurrent < current_now:
                        curMaxCurrent = current_now
                    current_per = int((curMaxCurrent * 1000) / target_current)
                    if current_per > 999: 
                        current_per = 999
                    if abs(current_now - target_current) < 2:
                        stableCount += 1
                        if stableCount >= stableDebounce:
                            out = True
                    else:
                        stableCount = 0
                    torque_per = int(maplist[current_per])
                    this_speed = int((max_speed * torque_per) / 1000)
                    print "this speed:%d, \n current_now:%d, \n current_per:%d, \n curMaxCurrent:%d"%(this_speed, current_now, current_per, curMaxCurrent)
                    if this_speed < 10: this_speed = 10
                    screw_dev.DeviceSetCurrentModeSpeedLimitation(this_speed)
                    print "Velocity limited at:%d"%screw_dev.DeviceGetMaxSpeedInCurrentMode()

                    while True and not out:
                        velocity_now = screw_dev.DeviceGetVelocityNow()

                        graphLock.acquire()
                        graphQueue.append(current_now)
                        if len(graphQueue) > GRAPH_X_RANGE:
                            graphQueue.pop(0)
                        velocityQueue.append(velocity_now)
                        if len(velocityQueue) > VELOCITY_X_RANGE:
                            velocityQueue.pop(0)
                        graphLock.release()

                        
                        if velocity_now <= this_speed:
                            break
                        elif velocity_now > this_speed:
                            offset = velocity_now - this_speed
                            if offset < 50:
                                break
                while target_current > 0:
                    target_current /= 2
                    screw_dev.DeviceSetTargetCurrent(target_current)
                screw_dev.DeviceSetTargetCurrent(0)
                
            elif "screwdriver1" in parameters[1]:
                print "step2"
                target_current = parameters[2]
                filename = parameters[3]
                if target_current is None:
                    raise ParamsInvalidException('No Target Current')
                if filename is None:
                    raise ParamsInvalidException('invalid network file')
                (screw_network, minx, maxx, srcfactor, dstfactor) = RestoreNetwork(filename, 1)
                if screw_network is None:
                    raise ParamsInvalidException('invalid network')

                target_current = int(target_current)
                print "Target Current:%d"%target_current
                print "Press Enter when you are ready"
                raw_input()

                graphCon.acquire()
                #graphics support
                GRAPH_X_RANGE = 1000
                graphQueue = [target_current for i in xrange(GRAPH_X_RANGE)]
                self.m_minx = 0
                self.m_maxx = 1000
                self.m_miny = 0
                self.m_maxy = target_current + 10

                VELOCITY_X_RANGE = 1000
                velocityQueue = [0 for i in xrange(VELOCITY_X_RANGE)]
                self.m_velminx = 0
                self.m_velmaxx = 1000
                self.m_velminy = 0
                self.m_velmaxy = 1000
                graphCon.notify()
                graphCon.release()

                dstfactor = 1.7 / (1000 - 0)
                srcfactor = 2.0 / ((target_current - 0) + 1)

                print dstfactor,srcfactor
                while True and screw_dev is not None:
                    current = ReadTorque()
#                    print "current is %d"%current
                    if current >= target_current:
                        screw_dev.DeviceSetTargetVelocity(0)
                        print "Current:%d is at or over target Current"%current
                        break
                    y = screw_network.Compute([float(current) * srcfactor - 1.0])[0]
#                    print y
                    velocity = (y + 0.85) / dstfactor + 0
                    screw_dev.DeviceSetTargetVelocity(int(velocity))
                    graphLock.acquire()
                    velocityQueue.append(velocity)
                    if len(velocityQueue) > VELOCITY_X_RANGE:
                        velocityQueue.pop(0)
                    graphLock.release()

                    

#main entry

simulate = True
screw_dev = InitDevControlNode()

GRAPH_X_RANGE = 1000
VELOCITY_X_RANGE = 1000
velocityQueue = None
graphQueue = None

parameters = sys.argv
fig = None
axes1 = None
axes2 = None
#axes2 = fig.add_subplot(2, 1, 2, xlim = (0, VELOCITY_X_RANGE), ylim = (0, 1000))

graphCon = threading.Condition()
graphLock = threading.RLock()

print "step1"
driver = ScrewDriver("screw")
graphCon.acquire()
driver.start()
graphCon.wait()
graphCon.release()

fig = plt.figure(1)
axes1 = fig.add_subplot(2, 1, 1, xlim=(driver.m_minx, driver.m_maxx), ylim=(driver.m_miny, driver.m_maxy))
axes2 = fig.add_subplot(2, 1, 2, xlim = (driver.m_velminx, driver.m_velmaxx), ylim = (driver.m_velminy, driver.m_velmaxy))

graphLock.acquire()
line, = axes1.plot(graphQueue)
line2, = axes2.plot(velocityQueue)
graphLock.release()

ani = animation.FuncAnimation(fig, update, init_func=init, interval=2*100)

plt.show()
driver.join()

