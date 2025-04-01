package moa.streams;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.capabilities.CapabilitiesHandler;
import moa.core.Example;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

import java.util.LinkedList;
import java.util.Random;

public class TrapezoidalMetaGenerator extends AbstractOptionHandler implements
        InstanceStream, CapabilitiesHandler {

    // GUI Base stream selection
    public ClassOption baseGeneratorOption = new ClassOption("baseGenerator",
            's',
            "Base stream.",
            ExampleStream.class,
            "generators.RandomTreeGenerator");

    // random seed
    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1, 0, Integer.MAX_VALUE);

    // Percentage of initial feature space
    public IntOption initialSpacePercentageOption = new IntOption("initialSpacePercentage",
            'I', "Initial space percentage.", 50, 0, 100);

    public IntOption spaceChangeWindowSizeOption = new IntOption("spaceChangeWindowSize",
            'w',
            "Window size for space changes.",
            10,
            1,
            100);

    public IntOption totalInstancesOption = new IntOption("totalInstances",
            't',
            "Total number of instances.",
            1000,
            1,
            Integer.MAX_VALUE);

    public MultiChoiceOption strategyOption = new MultiChoiceOption("strategy",
            'S', "Strategy for changing the feature space.",
            new String[] {"INCREMENT",
                    "REMOVAL", "HYPERBOLIC", "PARABOLIC", "CYCLE", "ORIGINAL"},
            new String[]{"Features will appear over time.",
                    "Features will disappear over time.",
                    "Features will disappear and reappear",
                    "Features will appear and disappear",
                    "Features will disappear but there is replacement",
                    "No streaming features"}, 0);

    // Original stream data
    protected ExampleStream baseStream;

    // Instances seen counter
    private long instancesSeen = 0;

    // List (index) of missing features
    LinkedList<Integer> indicesFaltantes = null;

    // Random number generator
    protected Random random = null;

    // Switch between appearing and disappearing
    protected boolean inverter = false;

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        this.baseStream = (ExampleStream) getPreparedClassOption(this.baseGeneratorOption);
        this.instancesSeen = 0;
        indicesFaltantes = new LinkedList<>();
        // Creating the instance of random generator number
        this.random = new Random(instanceRandomSeedOption.getValue());
        // Pre-selecting missing feature indexes
        int qtdAtributos = baseStream.getHeader().numAttributes() - 1;
        while (indicesFaltantes.size() != (int) Math.ceil((100 - (double) initialSpacePercentageOption.getValue())/100 * (baseStream.getHeader().numAttributes() - 1))){
            int posicao = this.random.nextInt(qtdAtributos);
            // if the position does not exist, is added to the missing features list
            if(!indicesFaltantes.contains(posicao)){
                indicesFaltantes.add(posicao);
            }
        }
    }

    @Override
    public Example<Instance> nextInstance() {
        // 1: select a complet instance from the base generator
        Example instanciaBase = baseStream.nextInstance();

        // 2: insert missing features into the instance
        limpaInstance(instanciaBase);

        // 3: updating the counter
        instancesSeen++;

        // 4: Checking if is time to change the feature space
        if (instancesSeen % Math.ceil(totalInstancesOption.getValue()*((double) spaceChangeWindowSizeOption.getValue()/100)) == 0) {
            if (strategyOption.getChosenLabel().equals("INCREMENT")) {
                features_increment();
            } else if (strategyOption.getChosenLabel().equals("REMOVAL")) {
                features_removal();
            } else if (strategyOption.getChosenLabel().equals("HYPERBOLIC")) {
                hyperbolic_features();
            } else if (strategyOption.getChosenLabel().equals("PARABOLIC")) {
                parabolic_features();
            } else if (strategyOption.getChosenLabel().equals("CYCLE")) {
                cycle_features();
            }

            String[] values = instanciaBase.toString().split(",");
            int nonNaNCount = 0;
            for (String value : values) {
                try {
                    Double.parseDouble(value);
                    nonNaNCount++;
                } catch (NumberFormatException e) {
                        // Ignore the exception as it means the value is not a number
                    }
                }
            //System.out.println("Number of non-NaN values: " + nonNaNCount + " instance: " + instancesSeen);
        }
        return instanciaBase;
    }

    private void features_increment() {
        // In order to add features, it is necessary to remove features from the missing feature list
        int qtdAdicionar = (int) Math.ceil((baseStream.getHeader().numAttributes() -1)*((double) spaceChangeWindowSizeOption.getValue()/100));
        for (int i = 0; i < qtdAdicionar; i++){
            if(indicesFaltantes.size() > 0){
                indicesFaltantes.remove();
            }
        }
    }

    private void features_removal() {
        // Increment the missing feature list to remove features
        int qtdRetirar = (int) Math.ceil((baseStream.getHeader().numAttributes() -1)*((double) spaceChangeWindowSizeOption.getValue()/100));
        this.random = new Random(instanceRandomSeedOption.getValue());
        int qtdAtributos = baseStream.getHeader().numAttributes() - 1;
        for (int i = 0; i < qtdRetirar; i++){
            while (indicesFaltantes.size() != baseStream.getHeader().numAttributes() - 1){
                int posicao = this.random.nextInt(qtdAtributos);
                if(!indicesFaltantes.contains(posicao)){
                    indicesFaltantes.add(posicao);
                    break;
                }
            }
        }
    }

    private void hyperbolic_features() {
        // The middle of the stream should have zero features. Works until space window of 25%, but with a small value of percentage it will reach zero feature before the middle
        int qtdAlterar = (int) Math.ceil((baseStream.getHeader().numAttributes() -1)*2*((double) spaceChangeWindowSizeOption.getValue()/100));
        // Halfway point of the stream
        int halfWayPoint = (int) Math.ceil(totalInstancesOption.getValue()/2);
        int desv = (int) Math.ceil(totalInstancesOption.getValue()*(double) spaceChangeWindowSizeOption.getValue()/200);

        // Do not change features in the middle of the Stream
        if(instancesSeen < halfWayPoint - desv || instancesSeen > halfWayPoint + desv) {
            // Features disappearing
            if (inverter == false) {
                if (indicesFaltantes.size() != baseStream.getHeader().numAttributes() - 1) {
                    this.random = new Random(instanceRandomSeedOption.getValue());
                    int qtdAtributos = baseStream.getHeader().numAttributes() - 1;
                    for (int i = 0; i < qtdAlterar; i++) {
                        while (indicesFaltantes.size() != baseStream.getHeader().numAttributes() - 1) {
                            int posicao = this.random.nextInt(qtdAtributos);
                            if (!indicesFaltantes.contains(posicao)) {
                                indicesFaltantes.add(posicao);
                                break;
                            }
                        }
                    }
                }
            }
            // Features appearing
            else {
                for (int i = 0; i < qtdAlterar; i++) {
                    if (indicesFaltantes.size() > 0) {
                        indicesFaltantes.remove();
                    }
                }
            }
        }
        // Switching strategy when the stream reached halfway point
        if (instancesSeen >= halfWayPoint - 1) {
            inverter = true;
        }

    }
    private void parabolic_features(){
        // The middle of the stream should have all features. Works until space window of 25%, but with a small value of percentage it will reach full feature before the middle
        int qtdAlterar = (int) Math.ceil((baseStream.getHeader().numAttributes() -1)*2*((double) spaceChangeWindowSizeOption.getValue()/100));
        // Halfway point of the stream
        int halfWayPoint = (int) Math.ceil(totalInstancesOption.getValue()/2);
        int desv = (int) Math.ceil(totalInstancesOption.getValue()*(double) spaceChangeWindowSizeOption.getValue()/200);

        // Do not change features in the middle of the Stream
        if(instancesSeen < halfWayPoint - desv || instancesSeen > halfWayPoint + desv) {
            // Features appearing
            if (inverter == false) {
                if (indicesFaltantes.size() > 0) {
                    for (int i = 0; i < qtdAlterar; i++) {
                        if (indicesFaltantes.size() > 0) {
                            indicesFaltantes.remove();
                        }
                    }
                }
            }
            // Features disappearing
            else {
                this.random = new Random(instanceRandomSeedOption.getValue());
                int qtdAtributos = baseStream.getHeader().numAttributes() - 1;
                for (int i = 0; i < qtdAlterar; i++) {
                    while (indicesFaltantes.size() != baseStream.getHeader().numAttributes() - 1) {
                        int posicao = this.random.nextInt(qtdAtributos);
                        if (!indicesFaltantes.contains(posicao)) {
                            indicesFaltantes.add(posicao);
                            break;
                        }
                    }
                }
            }
        }
        // Switching strategy when the stream reached halfway point
        if (instancesSeen >= halfWayPoint -1) {
            inverter = true;
        }
    }
    private void cycle_features(){
        // Starting with all features. Two features are removed and one appears until there are nothing left
        int qtdCiclar = (int) Math.ceil((baseStream.getHeader().numAttributes() -1)*((double) spaceChangeWindowSizeOption.getValue()/100));
        this.random = new Random(instanceRandomSeedOption.getValue());
        int qtdAtributos = baseStream.getHeader().numAttributes() - 1;
        // Features disappearing
        for (int i = 0; i < qtdCiclar; i++){
            while (indicesFaltantes.size() != baseStream.getHeader().numAttributes() - 1){
                int posicao = this.random.nextInt(qtdAtributos);
                if(!indicesFaltantes.contains(posicao)){
                    indicesFaltantes.add(posicao);
                    break;
                }
            }
        }
        // Features appearing
        for (int i = 0; i < qtdCiclar/2; i++){
            if(indicesFaltantes.size() > 0){
                indicesFaltantes.remove();
            }
        }
    }
    /**
     *
     * @param instnc
     * @return
     */
    void limpaInstance(Example<Instance> instnc){
        // Setting a missing value for each position from the missing feature list
        for(Integer posicao : indicesFaltantes){
            instnc.getData().setMissing(posicao);
        }
    }

    @Override
    public InstancesHeader getHeader() {
        return baseStream.getHeader();
    }

    @Override
    public long estimatedRemainingInstances() {
        return baseStream.estimatedRemainingInstances();
    }

    @Override
    public boolean hasMoreInstances() {
        return baseStream.hasMoreInstances();
    }

    @Override
    public boolean isRestartable() {
        return baseStream.isRestartable();
    }

    @Override
    public void restart() {

    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {

    }

}