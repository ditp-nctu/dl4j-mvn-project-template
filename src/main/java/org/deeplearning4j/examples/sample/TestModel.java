package org.deeplearning4j.examples.sample;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.Comparator;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
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
   String model_path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "lenetmnist.zip");

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
      try {
         mnistTest = new MnistDataSetIterator(batchSize, false, 12345);
         model = MultiLayerNetwork.load(new File(model_path), false);
         System.out.println(model.summary());
      } catch (IOException ex) {
         Logger.getLogger(TestModel.class.getName()).log(Level.SEVERE, null, ex);
      }
   }

   @Override
   public void draw() {

      if (mnistTest.hasNext()) {
         background(0);
         var input = mnistTest.next();
         var data0 = input.getFeatures().toFloatVector();
         for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
               var index = y * 28 + x;
               fill(data0[index] * 255);
               rect(x * dotSize, y * dotSize,
                       (x + 1) * dotSize, (y + 1) * dotSize);
            }
         }
         var array = model.activate(input.getFeatures(), Layer.TrainingMode.TEST).toFloatVector();
         var result = new StringBuilder();
         var guess = IntStream.range(0, array.length)
                 .peek(i -> result.append(String.format(" %d=%.3f ", i, array[i])))
                 .boxed()
                 .sorted(Comparator.comparing(i -> array[i], Comparator.reverseOrder()))
                 .findFirst().get();
         var labels = input.getLabels().toFloatVector();
         var answer = IntStream.range(0, labels.length)
                 .filter(i -> labels[i] > 0)
                 .findFirst().getAsInt();
         fill(guess == answer ? 255 : Color.RED.getRGB());
         text(answer, dotSize, dotSize * 4);
         System.out.printf("%s [%s]\n",
                 result,
                 "A=" + answer + (answer == guess ? "" : ",G=" + guess));
      } else {
         exit();
      }
   }

   public static void main(String[] args) {

      PApplet.main(TestModel.class.getName());
   }
}
