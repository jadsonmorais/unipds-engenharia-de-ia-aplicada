// import tf from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

async function TrainModel(inputXs, outputYs) {
    const model = tf.sequential()
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu'}))
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))
    model.compile({
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(
                //     `Epoch: ${epoch}: loss = ${log.loss}`
                // )
            }
        }
    )

    return model
}

async function  predict(model, pessoa){
    // transformar array js para o tensor
    const tfInput = tf.tensor2d(pessoa)

    // faz a predicao será um vetor de 3 possiblidades
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    // console.log(predArray)
    return predArray[0].map((prob, index) => ({prob, index}))
}

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print();
outputYs.print();

const model = await TrainModel(inputXs, outputYs)

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: "Curitiba"}

// normalizando a idade da pessoa
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.12

const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        0, // cor azul
        0, // cor vermelho
        1, // cor verde
        0, // localização Sao paulo
        0, // localização Rio
        1 // localização Curitiba
    ]
]
// await predict(model, pessoaTensorNormalizado)

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')

console.log(results)