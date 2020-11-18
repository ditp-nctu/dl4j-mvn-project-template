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
import java.net.URI;
import java.net.URISyntaxException;
import java.util.AbstractMap.SimpleEntry;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
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

    Map<String, Integer> inputs;
    Iterator<SimpleEntry<PImage, Integer>> source;

    @Override
    public void setup() {

        colorMode(RGB);
        frameRate(1.5f);
        strokeWeight(0.1f);
        stroke(100f);
        textSize(dotSize * 4);

        inputs = new HashMap<>();
        inputs.put("num_0.png", 0);
        inputs.put("num_1.png", 1);
        inputs.put("num_2.png", 2);
        inputs.put("num_3.png", 3);
        inputs.put("num_4.png", 4);
        inputs.put("num_5.png", 5);
        inputs.put("num_6.png", 6);
        inputs.put("num_7.png", 7);
        inputs.put("num_8.png", 8);
        inputs.put("num_9.png", 9);
        try {
            model = MultiLayerNetwork.load(new File(path), false);
            System.out.println(model.summary());
            source = inputs.entrySet().stream()
                    .map(e -> new SimpleEntry<>(loadImage(filenameToURI(e.getKey()).getPath()), e.getValue()))
                    .collect(Collectors.toList())
                    .iterator();
        } catch (Exception ex) {
            Logger.getLogger(TestModel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static URI filenameToURI(String filename) {

        try {
            return Guess.class.getResource("/numbers/" + filename).toURI();
        } catch (URISyntaxException ex) {
            Logger.getLogger(Guess.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

    @Override
    public void draw() {

        if (source.hasNext()) {
            background(0);
            Entry<PImage, Integer> input = source.next();
            int[] data = input.getKey().pixels;
            float[] data0 = new float[data.length];
            for (int i = 0; i < data.length; i++) {
                data0[i] = (float) (data[i] & 0x00ffff) / 0xffff;
            }
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int index = y * 28 + x;
                    fill(data0[index] * 255);
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
            noLoop();
        }
    }

    public static void main(String[] args) {
        PApplet.main(Guess.class.getName());
    }
}
