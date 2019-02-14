/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.harun.javadl;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;

/**
 *
 * @author HARUN
 */
public class Program {

    public static void main(String[] args) {

        int batchSize = 16; // how many examples simultaneously train in the network
        EmnistDataSetIterator.Set emnistSet = EmnistDataSetIterator.Set.BALANCED;
        EmnistDataSetIterator emnistTrain = null;
        EmnistDataSetIterator emnistTest = null;
        try {
            emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
            emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);
        } catch (Exception e) {
            e.printStackTrace();
        }
        MultiLayerConfiguration multiLayerConf = new NeuralNetConfiguration.Builder()
                .seed(123).learningRate(0.1).iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(100).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(1, new OutputLayer.Builder().nIn(100).nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SIGMOID).build())
                .pretrain(false).backprop(true)
                .build();
        System.out.println(multiLayerConf.toJson());

        MultiLayerNetwork network = new MultiLayerNetwork(multiLayerConf);
        network.init();
        network.addListeners(new ScoreIterationListener(5));
        network.fit(emnistTrain);
        // network.fit(new MultipleEpochsIterator(2, emnistTrain));
        
        Evaluation eval = network.evaluate(emnistTest);
        eval.accuracy();
        eval.precision();
        eval.recall();
        System.out.println(eval.stats());
    }
}
