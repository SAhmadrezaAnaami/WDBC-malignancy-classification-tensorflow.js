const tf = require("@tensorflow/tfjs")

async function run(){
    const trainURL ="http://localhost:8000/wdbc-train.csv";
    const trainData = tf.data.csv(
        trainURL,
        {
            columnConfigs:{
                diagnosis:{
                    isLabel:true
                }
            }
        }
    )
    const convertedTrainData = trainData.map(
        ({xs, ys}) => 
        {
            return{ xs: Object.values(xs), ys: Object.values(ys)};
        }
    ).batch(10);
    
    const testURL ="http://localhost:8000/wdbc-test.csv";
    const testData = tf.data.csv(
        testURL,
        {
            columnConfigs:{
                diagnosis:{
                    isLabel:true
                }
            }
        }
    )
    const convertedTestData = testData.map(
        ({xs, ys}) => 
        {
            return{ xs: Object.values(xs), ys: Object.values(ys)};
        }
    ).batch(10);

    const numOfFeatures = (await trainData.columnNames()).length - 1

    const model = tf.sequential();
    
    model.add(tf.layers.dense({inputShape: [numOfFeatures], activation: "relu", units: 32}))
    model.add(tf.layers.dense({activation: "relu", units: 64}));
    model.add(tf.layers.dense({activation: "relu", units: 32}));
    model.add(tf.layers.dense({activation: "sigmoid", units: 1}));

    model.compile({
            loss: "binaryCrossentropy",
            optimizer: tf.train.rmsprop(0.01),
            metrics: ['accuracy']
    })
    
    await model.fitDataset(
        convertedTrainData, 
        {
            epochs:50,
            validationData: convertedTestData,
            callbacks:
            {
                onEpochEnd: async(epoch, logs) =>
                {
                    console.log("Epoch: " + epoch + " Loss: " + logs.loss + " Accuracy: " + logs.acc);
                }
            }
        });
    
    const testVal0 = tf.tensor2d([-1.482236314,-0.8816887618,-1.444366388,-1.153859693,-0.9781741207,-0.5222354048,-0.5182272828,-0.6453120827,0.4406963519,0.5200127253,0.4772846793,-0.0182513709,0.7132123259,-0.2119558294,1.382346477,0.2245085961,0.02147345557,0.5794969791,0.9581369744,0.1778814476,-1.284517963,-1.286321853,-1.234377841,-0.9887989613,-1.369823868,-0.8307898449,-0.8679323628,-1.024194661,-0.7472083943,-0.5159744877], [1, numOfFeatures]);
    const testVal1 = tf.tensor2d([0.2881225321,2.565267213,0.2014188421,0.1835828701,-0.9585864666,-1.135506599,-0.5227308617,-0.5524577898,0.1556896364,-1.428211917,0.2308560459,0.7704951191,0.1069770387,0.1068498824,-0.06069042379,-0.6170903088,-0.2118587849,-0.4344719102,1.341649246,-0.7519872885,0.2054999116,1.865998536,0.09070105662,0.08571921809,-0.7813513189,-1.001705405,-0.5656025799,-0.7463359761,0.522612409,-1.227436431], [1, numOfFeatures]);
    const prediction0 = (await model.predict(testVal0).data());
    console.log(Math.round(prediction0))
    const prediction1 = (await model.predict(testVal1).data());
    console.log(Math.round(prediction1))
}

run();