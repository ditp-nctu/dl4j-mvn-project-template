package org.deeplearning4j.examples.sample;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import processing.core.PApplet;

/**
 *
 * @author Jonathan Chang, Chun-yien <ccy@musicapoetica.org>
 */
public class TestModel extends PApplet {

    int batchSize = 1; // Test batch size;
    int dotSize = 10;
    MultiLayerNetwork model;
    DataSetIterator mnistTest;
    String path;

    @Override
    public void settings() {

        size(dotSize * 28, dotSize * 28);
    }

    @Override
    public void setup() {

        colorMode(RGB);
        frameRate(1.5f);
        strokeWeight(0.1f);
        stroke(100f);
        textSize(dotSize * 4);
        path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "lenetmnist.zip");
        try {
            mnistTest = new MnistDataSetIterator(batchSize, false, 12345);
            model = MultiLayerNetwork.load(new File(path), false);
            System.out.println(model.summary());
        } catch (IOException ex) {
            Logger.getLogger(TestModel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void draw() {

        background(0);
        if (mnistTest.hasNext()) {
            var input = mnistTest.next();
            var data0 = input.asList().get(0).getFeatures().toFloatVector();
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    var index = y * 28 + x;
                    var point = data0[index] * 255;
                    if (point > 255) throw new RuntimeException();
                    fill(point);
                    rect(x * dotSize, y * dotSize,
                            (x + 1) * dotSize, (y + 1) * dotSize + dotSize);
                }
            }
            var array = model.activate(input.getFeatures(), Layer.TrainingMode.TEST).toFloatMatrix()[0];
            var result = "";
            var guess = 0;
            var highest = 0f;
            for (int i = 0; i < array.length; i++) {
                float f = array[i];
                result += String.format(" %d=%.3f ", i, f);
                if (f > highest) {
                    highest = f;
                    guess = i;
                }
            }
            var labels = input.getLabels().toFloatVector();
            var answer = 0;
            for (int i = 0; i < labels.length; i++) {
                if (labels[i] > 0) {
                    answer = i;
                    break;
                }
            }
            fill(guess == answer ? 255 : Color.RED.getRGB());
            text(answer, dotSize, dotSize * 4);
            System.out.printf("%s %s\n",
                    result,
                    (answer == guess) ? "" : "[A=" + answer + ",G=" + guess + "]");
        } else {
            var result = model.evaluate(mnistTest);
            System.out.println(result);
            noLoop();
        }
    }

    public static void main(String[] args) {
        PApplet.main(TestModel.class.getCanonicalName());
    }
}
