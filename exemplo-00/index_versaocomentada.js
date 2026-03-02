// Importa a biblioteca TensorFlow.js. O '*' importa todas as funcionalidades como o objeto 'tf'.
import * as tf from '@tensorflow/tfjs';

/**
 * Função assíncrona para criar, configurar e treinar o modelo de rede neural.
 * @param {tf.Tensor} inputXs - Tensores de entrada (características das pessoas).
 * @param {tf.Tensor} outputYs - Tensores de saída (rótulos/categorias esperadas).
 */
async function TrainModel(inputXs, outputYs) {
    // 1. Cria um modelo sequencial: as camadas são conectadas uma após a outra em ordem.
    const model = tf.sequential();

    // 2. Adiciona a Camada Oculta (Hidden Layer):
    // inputShape: [7] -> Define que cada dado de entrada tem 7 números (idade + cores + locais).
    // units: 80 -> A camada possui 80 neurônios para aprender padrões complexos.
    // activation: 'relu' -> Função que ajuda a rede a aprender relações não-lineares.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // 3. Adiciona a Camada de Saída (Output Layer):
    // units: 3 -> Prevemos 3 categorias (premium, medium, basic).
    // activation: 'softmax' -> Transforma a saída em probabilidades que somam 100% (1.0).
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // 4. Compilação do Modelo:
    // optimizer: 'adam' -> Algoritmo inteligente que ajusta os pesos para reduzir o erro.
    // loss: 'categoricalCrossentropy' -> Função de perda ideal para classificação múltipla.
    // metrics: ['accuracy'] -> Monitora a porcentagem de acertos durante o treino.
    model.compile({
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // 5. Treinamento propriamente dito:
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,      // 0 = silencioso, 1 = mostra progresso no console.
            epochs: 100,     // O modelo passará pelos dados 100 vezes.
            shuffle: true,   // Embaralha os dados em cada época para evitar vícios na ordem.
            callbacks: {}    // Espaço para funções que executam durante o treino (ex: logs).
        }
    );

    return model;
}

/**
 * Função para realizar a predição baseada em um modelo já treinado.
 */
async function predict(model, pessoa) {
    // Transforma o array JavaScript simples em um Tensor 2D (formato que o TF entende).
    const tfInput = tf.tensor2d(pessoa);

    // model.predict retorna um Tensor com as probabilidades para cada classe.
    const pred = model.predict(tfInput);
    
    // Converte o Tensor de volta para um array JavaScript comum de forma assíncrona.
    const predArray = await pred.array();
    
    // Mapeia os resultados para associar o valor da probabilidade ao seu índice original.
    return predArray[0].map((prob, index) => ({ prob, index }));
}

// --- PREPARAÇÃO DOS DADOS ---

// Dados de entrada normalizados (0 a 1) e One-Hot Encoded (categorias transformadas em 0 e 1).
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Ex: Erick (Idade normalizada, Cor Azul, Local SP)
    [0,    0, 1, 0, 0, 1, 0], // Ex: Ana
    [1,    0, 0, 1, 0, 0, 1]  // Ex: Carlos
];

const labelsNomes = ["premium", "medium", "basic"];

// Labels de saída: o que queremos que o modelo aprenda a responder para os dados acima.
const tensorLabels = [
    [1, 0, 0], // premium
    [0, 1, 0], // medium
    [0, 0, 1]  // basic
];

// Converte os arrays de treino em Tensores 2D.
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// Exibe os tensores no console para conferência visual da estrutura.
inputXs.print();
outputYs.print();

// Inicia o processo de treinamento e aguarda a finalização.
const model = await TrainModel(inputXs, outputYs);

// --- TESTE DE PREDIÇÃO ---

// Dados de uma nova pessoa (Zé) para testar se o modelo aprendeu.
const pessoaTensorNormalizado = [
    [
        0.2, // idade (normalizada entre o min/max do treino)
        0,   // não é azul
        0,   // não é vermelho
        1,   // é verde
        0,   // não é SP
        0,   // não é Rio
        1    // é Curitiba
    ]
];

// Executa a predição para o "Zé".
const predictions = await predict(model, pessoaTensorNormalizado);

// Formatação do resultado para o usuário:
const results = predictions
    .sort((a, b) => b.prob - a.prob) // Coloca a maior probabilidade no topo.
    .map(p => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`) // Formata como texto.
    .join('\n');

console.log("Resultado da Predição:\n" + results);