/*
 *    DSARFHoeffdingTree.java
 * 
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

package moa.classifiers.trees;

import com.github.javacliparser.IntOption;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.core.Utils;
import com.yahoo.labs.samoa.instances.Instance;

import java.util.ArrayList;
/**
 * DSAdaptive Random Forest Hoeffding Tree.
 * 
 * <p>DSAdaptive Random Forest Hoeffding Tree. This is the base model for the
 * Adaptive Random Forest ensemble learner 
 * (See moa.classifiers.meta.AdaptiveRandomForest.java). This Hoeffding Tree
 * includes a subspace size k parameter, which defines the number of randomly 
 * selected features to be considered at each split. </p>
 * 
 * <p>See details in:<br> Heitor Murilo Gomes, Albert Bifet, Jesse Read, 
 * Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, 
 * Talel Abdessalem. Adaptive random forests for evolving data stream classification. 
 * In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.</p>
 *
 * @author Heitor Murilo Gomes (heitor_murilo_gomes at yahoo dot com dot br)
 * @version $Revision: 1 $
 */
public class DSARFHoeffdingTree extends HoeffdingTree {

    private static final long serialVersionUID = 1L;
    
    public IntOption subspaceSizeOption = new IntOption("subspaceSizeSize", 'k',
            "Number of features per subset for each node split. Negative values = #features - k", 
            2, Integer.MIN_VALUE, Integer.MAX_VALUE);
    
    @Override
    public String getPurposeString() {
        return "Adaptive Random Forest Hoeffding Tree for data streams. "
                + "Base learner for AdaptiveRandomForest.";
    }

    public static class RandomLearningNode extends ActiveLearningNode {
        
        private static final long serialVersionUID = 1L;

        protected int[] listAttributes;

        protected int numAttributes;
        
        public RandomLearningNode(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations);
            this.numAttributes = subspaceSize;
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {            
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());

            if (this.listAttributes == null) {
                // 1) Collect all non-NaN attribute indices (excluding class attribute)
                ArrayList<Integer> n_Attributes = new ArrayList<>();
                for (int j = 0; j < inst.numAttributes(); ++j) {
                    if (j == inst.classIndex()) {
                        continue; // skip class attribute
                    }
                    double val = inst.value(j);
                    if (!Double.isNaN(val)) {
                        n_Attributes.add(j);
                    }
                }
                int available = n_Attributes.size();
                // 2) Determine actual subspace size (cannot exceed available)
                int k = Math.min(this.numAttributes, available);
                this.listAttributes = new int[k];

                // 3) Randomly pick k unique attributes from filtered list
                for (int j = 0; j < k; j++) {
                    boolean isUnique = false;
                    while (!isUnique) {
                        int idx = ht.classifierRandom.nextInt(available);
                        int att = n_Attributes.get(idx);
                        isUnique = true;
                        for (int i = 0; i < j; i++) {
                            if (this.listAttributes[i] == att) {
                                isUnique = false;
                                break;
                            }
                        }
                        if (isUnique) {
                            this.listAttributes[j] = att;
                        }
                    }
                }
            }

            // 4) Update observers only on selected non-NaN attributes
            for (int j = 0; j < this.listAttributes.length - 1; j++) {
                int i = this.listAttributes[j];
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal()
                            ? ht.newNominalClassObserver()
                            : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }
        }
    }

    public static class LearningNodeNB extends RandomLearningNode {

        private static final long serialVersionUID = 1L;

        public LearningNodeNB(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations, subspaceSize);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
                return NaiveBayes.doNaiveBayesPrediction(inst,
                        this.observedClassDistribution,
                        this.attributeObservers);
            }
            return super.getClassVotes(inst, ht);
        }

        @Override
        public void disableAttribute(int attIndex) {
            // should not disable poor atts - they are used in NB calc
        }
    }

    public static class LearningNodeNBAdaptive extends LearningNodeNB {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;

        public LearningNodeNBAdaptive(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations, subspaceSize);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            int trueClass = (int) inst.classValue();
            if (this.observedClassDistribution.maxIndex() == trueClass) {
                this.mcCorrectWeight += inst.weight();
            }
            if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers)) == trueClass) {
                this.nbCorrectWeight += inst.weight();
            }
            super.learnFromInstance(inst, ht);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (this.mcCorrectWeight > this.nbCorrectWeight) {
                return this.observedClassDistribution.getArrayCopy();
            }
            return NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers);
        }
    }

    public DSARFHoeffdingTree() {
        this.removePoorAttsOption = null;
    }
    
    @Override
    protected LearningNode newLearningNode(double[] initialClassObservations) {
        LearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        if (predictionOption == 0) { //MC
            ret = new RandomLearningNode(initialClassObservations, this.subspaceSizeOption.getValue());
        } else if (predictionOption == 1) { //NB
            ret = new LearningNodeNB(initialClassObservations, this.subspaceSizeOption.getValue());
        } else { //NBAdaptive
            ret = new LearningNodeNBAdaptive(initialClassObservations, this.subspaceSizeOption.getValue());
        }
        return ret;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}