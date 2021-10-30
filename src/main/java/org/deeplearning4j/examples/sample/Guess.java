/*
 * Copyright 2020 Jonathan Chang, Chun-yien <ccy@musicapoetica.org>.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.deeplearning4j.examples.sample;

import java.awt.Color;
import java.io.File;
import java.io.InputStream;
import java.util.AbstractMap.SimpleEntry;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import processing.core.PApplet;
import processing.core.PImage;

/**
 *
 * @author Jonathan Chang, Chun-yien <ccy@musicapoetica.org>
 */
public class Guess extends TestModel {

   Properties inputs = new Properties();
   Iterator<SimpleEntry<String, Integer>> source;

   static final String INPUT_LIST = "/input_list.properties";

   @Override
   public void setup() {

      colorMode(RGB);
      frameRate(1.5f);
      strokeWeight(0.1f);
      stroke(100f);
      textSize(dotSize * 4);

      try ( InputStream input_list = this.getClass()
              .getResourceAsStream(INPUT_LIST)) {
         model = MultiLayerNetwork.load(new File(model_path), false);
         inputs.load(input_list);
      } catch (Exception ex) {
         Logger.getLogger(TestModel.class.getName()).log(Level.SEVERE, null, ex);
      }
      System.out.println(model.summary());
      source = inputs.entrySet().stream()
              .map(e -> new SimpleEntry<>(e.getKey().toString(), Integer.valueOf(e.getValue().toString().trim())))
              .iterator();
   }

   public String getImagePath(String filename) {

      return this.getClass().getResource("/numbers/" + filename).getPath();
   }

   @Override
   public void draw() {

      if (source.hasNext()) {
         background(0);
         Entry<String, Integer> input = source.next();
         PImage image = loadImage(getImagePath(input.getKey()));
         float[] data0 = new float[28 * 28];
         for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
               int index = y * 28 + x;
               int point = image.get(x, y);
               data0[index] = (float) (point & 0x00ffff) / 0xffff;
               fill(point);
               rect(x * dotSize, y * dotSize,
                       (x + 1) * dotSize, (y + 1) * dotSize);
            }
         }
         float[] array = model.activate(new NDArray(data0), Layer.TrainingMode.TEST).toFloatVector();
         StringBuilder result = new StringBuilder();
         int guess = IntStream.range(0, array.length)
                 .peek(i -> result.append(String.format(" %d=%.3f ", i, array[i])))
                 .boxed()
                 .sorted(Comparator.comparing(i -> array[i], Comparator.reverseOrder()))
                 .findFirst().get();
         int answer = input.getValue();
         fill(answer == guess ? 255 : Color.RED.getRGB());
         text(answer, dotSize, dotSize * 4);
         System.out.printf("%s [%s]\n",
                 result,
                 "A=" + answer + (answer == guess ? "" : ",G=" + guess));
      } else {
         this.exit();
      }
   }

   public static void main(String[] args) {

      PApplet.main(Guess.class.getName());
   }
}
