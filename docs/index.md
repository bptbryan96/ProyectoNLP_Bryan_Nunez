# Clasificaciónn de frases del portal de compras públicas del Ecuador en acusatorio y NO acusatorio

## Introducciónn

En el ámbito de las compras públicas en Ecuador, la fase de preguntas y respuestas entre proveedores y entidades gubernamentales se presenta como una ventana hacia las dinámicas de interacción en estos procesos. Esta etapa es crucial, ya que permite identificar preocupaciones significativas que pueden incluir acusaciones de favoritismo o corrupción. Tales acusaciones, si no se gestionan correctamente, pueden socavar la integridad de todo el proceso de licitación.
El desafío principal radica en la capacidad de clasificar de manera eficiente y precisa estas preguntas en categorías de acusatorias y no acusatorias. Actualmente, la clasificación manual no solo es ineficiente, sino también susceptible a sesgos y errores. En este contexto, la aplicación de tecnologías avanzadas de procesamiento de lenguaje natural, específicamente el uso de modelos de lenguaje como RNN, LSTM o GRU, promete mejorar significativamente la precisión y la velocidad de esta clasificación .
Este estudio propone el desarrollo de un sistema automatizado que utiliza las 3 tecnicas anteriores mencionadas para la clasificación textual y ChatGPT para la generación de datos de entrenamiento adicionales, con el objetivo de crear un modelo robusto capaz de discernir entre comentarios acusatorios y no acusatorios en el contexto de las licitaciones públicas. La implementación de tal sistema no solo podría mejorar la eficiencia del proceso de revisión de preguntas y respuestas, sino también fortalecer la transparencia y la equidad en las compras públicas .

## Augmento de datos

Inicialmente, el conjunto de datos completo fue dividido en dos segmentos principales: un 80% (4004frases) destinado para el entrenamiento y un 20% (1001 frases) reservado para la prueba. Teniendo 3886 frases de la clase 'NO Acusatorio' y 118 frases de la clase 'Acusatoria'
Dado el desequilibrio notable en la distribución de clases observado en el dataset, con una predominancia de frases no acusatorias, se implementó un proceso de aumento de datos para mejorar este desequilibrio. 

<div style="text-align: center;">
  <img src="./images/distriOriginal.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

Se utilizó un metodo de promting y el modelo GPT-4o-mini de OpenAI, generamos 5027 nuevas frases acusatorias. Este enfoque aumentó significativamente el número de frases acusatorias, ayudando a mejorar el entrenamiento del modelo.
Después de aplicar el aumento de datos, se obtuvo un total de 9031 frases en total (3886 frases para la clase “No Acusatoria”, y 5145 frases para la clase “Acusatoria”).

<div style="text-align: center;">
  <img src="./images/distriDA.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

## Preprocesamiento de datos
### 1. Tokenización
Se realizó la tokenización del texto con el fin de segmentarlo en palabras individuales o "tokens". Para ello, se utilizó el método word_tokenize, el cual permite dividir cada frase en una secuencia de palabras. Adicionalmente, se convirtió todo el texto a minúsculas para normalizar las entradas y evitar que diferencias en el uso de mayúsculas introdujeran ruido en el procesamiento.

### 2. Eliminación de Stop Words
Se procedió a eliminar las stop words del texto, que son palabras funcionales de alta frecuencia que no aportan información significativa para el análisis de contenido. El conjunto de stop words utilizado fue el proporcionado por la biblioteca NLTK en español. Este paso es crucial para reducir el volumen de información irrelevante y optimizar el desempeño del modelo, dado que palabras como "el", "de", y "y" no contribuyen significativamente a la comprensión semántica del texto.

### 3. Stemming (Raíces de Palabras)
Para reducir las palabras a su forma raíz, se aplicó la técnica de stemming mediante el uso del algoritmo de Porter (PorterStemmer). Este proceso permitió reducir variaciones morfológicas de una palabra a una raíz común, facilitando el tratamiento de palabras con formas flexionadas de manera uniforme. Aunque el stemming puede generar raíces no léxicas, su uso es útil para reducir la dimensionalidad del vocabulario sin perder generalidad en las palabras claves.

### 4. Lematización
Se aplicó la lematización para transformar las palabras en su forma base o lema, utilizando el algoritmo WordNetLemmatizer. A diferencia del stemming, este método asegura que las palabras se conviertan a una forma válida en el idioma, mejorando la precisión en comparación con el simple recorte de sufijos. La lematización permite conservar el significado completo de las palabras, lo que es fundamental cuando se trabaja con textos que requieren análisis semántico más detallado.

<div style="display: flex; justify-content: space-around; align-items: center;">

  <div style="text-align: center;">
    <img src="./images/nubeOriginal.png" alt="Imagen 1" style="width: 600px; height: auto;">
  </div>

  <div style="text-align: center;">
    <img src="./images/nubePrepo.png" alt="Imagen 2" style="width: 600px; height: auto;">
  </div>

</div>


## Entrenamiento
### RNN
#### 1. Definición de la Arquitectura RNN
Se implementó una red neuronal recurrente (RNN) denominada SentimentRNN, con el propósito de capturar las dependencias secuenciales en las representaciones de texto para el análisis de sentimientos. La arquitectura incluye una capa de embeddings (nn.Embedding), que mapea cada palabra en un vector de dimensión fija, seguido de una capa recurrente simple (nn.RNN) que modela la relación entre palabras en secuencias. La capa final es una capa totalmente conectada (nn.Linear) que convierte la última salida de la secuencia en una predicción sobre las dos clases de salida: acusatoria o no acusatoria. Esta configuración permite que el modelo aprenda patrones temporales en las secuencias de texto, lo cual es fundamental en el procesamiento de lenguaje natural.

#### 2. Entrenamiento del Modelo
El proceso de entrenamiento del modelo SentimentRNN se realizó utilizando el optimizador Adam (optim.Adam), conocido por su capacidad de ajustarse dinámicamente a diferentes magnitudes de gradiente, con una tasa de aprendizaje inicial de 0.001. La función de pérdida seleccionada fue la CrossEntropyLoss, adecuada para problemas de clasificación multiclase. A lo largo de 50 épocas, se registraron tanto la pérdida promedio como la precisión macro-promediada (MulticlassAccuracy), evaluada sobre el conjunto de entrenamiento. Esta métrica proporcionó una visión balanceada del desempeño del modelo al considerar ambas clases de manera equitativa.

### LSTM
#### 1. Definición de la Arquitectura LSTM
Se implementó un modelo de red neuronal recurrente con Long Short-Term Memory (LSTM), denominado SentimentLSTM, con el objetivo de modelar las relaciones secuenciales entre palabras en el contexto del análisis de sentimientos. La arquitectura consta de tres componentes principales: una capa de embeddings (nn.Embedding) que transforma las palabras en vectores de representación continua, una capa LSTM (nn.LSTM) que captura dependencias a largo plazo en las secuencias, y una capa totalmente conectada (nn.Linear) que mapea la salida de la LSTM a las clases de predicción (acusatoria o no acusatoria). La salida final se obtiene tomando el último estado oculto de la secuencia, lo que permite que el modelo integre información de toda la secuencia antes de generar una predicción.

#### 2. Entrenamiento del Modelo
El entrenamiento del modelo SentimentLSTM se llevó a cabo utilizando el optimizador Adam (optim.Adam), con una tasa de aprendizaje de 0.001. Se empleó la función de pérdida CrossEntropyLoss, adecuada para la clasificación multiclase. El proceso de entrenamiento incluyó 50 épocas, durante las cuales se midió tanto la pérdida media como la precisión macro-promediada a nivel de cada época. Se utilizó la métrica MulticlassAccuracy para calcular la precisión general del modelo en la tarea de clasificación binaria, brindando una visión equilibrada del rendimiento en ambas clases (acusatoria y no acusatoria).

### GRU
#### 1. Definición de la Arquitectura GRU
Se diseñó un modelo de red neuronal recurrente basado en Gated Recurrent Units (GRU), denominado SentimentGRU, con el fin de capturar las dependencias temporales en el texto para el análisis de sentimientos. La arquitectura del modelo incluye una capa de embeddings (nn.Embedding) para transformar cada palabra en una representación vectorial, una capa GRU (nn.GRU) que modela la información secuencial en las frases, y una capa totalmente conectada (nn.Linear) que genera la predicción final. Al igual que en otros modelos recurrentes, la última salida de la secuencia se toma como representación final para realizar la clasificación, permitiendo al modelo aprender las dependencias a lo largo de toda la secuencia de texto.

#### 2. Entrenamiento del Modelo
El entrenamiento del modelo SentimentGRU se llevó a cabo utilizando el optimizador Adam (optim.Adam), con una tasa de aprendizaje inicial de 0.001, y la función de pérdida CrossEntropyLoss, optimizada para tareas de clasificación multiclase. Durante el proceso de entrenamiento, que abarcó 50 épocas, se calcularon tanto la pérdida como la precisión en cada iteración. La precisión fue medida mediante la métrica MulticlassAccuracy para proporcionar una visión global del rendimiento del modelo sobre las dos clases objetivo (acusatoria y no acusatoria). Este enfoque permitió un seguimiento detallado del progreso del modelo en la tarea de clasificación binaria.

## RESULTADOS
### Resporte de Clasificación
#### 1. RNN (Red Neuronal Recurrente Simple)
Análisis: El modelo RNN tiene un rendimiento muy pobre en la detección de la clase "No Acusatoria" con un recall de solo 0.22, lo que sugiere que no está identificando correctamente muchas instancias de esta clase. Sin embargo, para la clase "Acusatoria", tiene un recall de 0.83, aunque la precisión es extremadamente baja (0.03), lo que indica un alto número de falsos positivos. El F1-score es muy bajo para ambas clases, lo que refleja un desempeño deficiente en general.

<div style="text-align: center;">
  <img src="./images/RNN_REPORTE.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

#### 2. LSTM (Long Short-Term Memory)
Análisis: El modelo LSTM tiene una mejor precisión en general (0.89), pero sigue teniendo problemas para detectar la clase "Acusatoria" con una precisión de solo 0.06 y un recall de 0.21. El F1-score es bajo para la clase "Acusatoria", lo que indica que, aunque el modelo mejora significativamente la detección de la clase "No Acusatoria", aún tiene dificultades para reconocer correctamente las instancias de la clase "Acusatoria".

<div style="text-align: center;">
  <img src="./images/LSTM_REPORTE.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

#### 3. GRU (Gated Recurrent Unit)
Análisis: El modelo GRU presenta un rendimiento general superior, con una alta precisión (0.96) y F1-score para la clase "No Acusatoria" (0.98). Aunque la clase "Acusatoria" sigue siendo un desafío, el modelo logra una mejora en precisión (0.21) y F1-score (0.21), lo que lo hace el mejor de los tres modelos en términos de balance entre ambas clases.

<div style="text-align: center;">
  <img src="./images/GRU_REPORTE.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

### Pérdida y presición durante el entrenamiento de los modelos
#### 1. RNN (Red Neuronal Recurrente Simple)
Pérdida vs Época: La pérdida comienza en torno a 0.68 y muestra una tendencia a aumentar hasta un pico cercano a 0.73 en las primeras 15 épocas. Luego de este aumento, la pérdida desciende gradualmente pero se mantiene algo inestable a lo largo del resto del entrenamiento, con fluctuaciones entre 0.69 y 0.70.
Precisión vs Época: La precisión muestra una considerable inestabilidad, comenzando en 0.51 y fluctuando entre 0.48 y 0.53 durante todo el proceso. No hay una tendencia clara de mejora a lo largo de las épocas, lo que sugiere que el modelo RNN no logra capturar patrones suficientes para mejorar consistentemente.

<div style="display: flex; justify-content: space-around; align-items: center;">

  <div style="text-align: center;">
    <img src="./images/RNN_LOSS.png" alt="Imagen 1" style="width: 600px; height: auto;">
  </div>

  <div style="text-align: center;">
    <img src="./images/RNN_ACC.png" alt="Imagen 2" style="width: 600px; height: auto;">
  </div>

</div>

#### 2. LSTM (Long Short-Term Memory)
Pérdida vs Época: La pérdida comienza cerca de 0.70 y desciende rápidamente en las primeras 10 épocas, alcanzando un valor cercano a 0.2. Tras este descenso inicial, la pérdida sigue una tendencia a la baja pero de forma mucho más gradual, con una relativa estabilidad en torno a 0.15 a partir de la época 20.
Precisión vs Época: La precisión en LSTM también mejora de manera más significativa en comparación con RNN. Comienza en 0.50 y sube rápidamente a más de 0.90 en las primeras 10 épocas. Aunque hay algunas fluctuaciones menores, la precisión se mantiene cerca de 0.90-0.95 durante el resto del entrenamiento, lo que indica que el modelo está capturando adecuadamente las dependencias temporales.

<div style="display: flex; justify-content: space-around; align-items: center;">

  <div style="text-align: center;">
    <img src="./images/LSTM_LOSS.png" alt="Imagen 1" style="width: 600px; height: auto;">
  </div>

  <div style="text-align: center;">
    <img src="./images/LSTM_ACC.png" alt="Imagen 2" style="width: 600px; height: auto;">
  </div>

</div>

#### 3. GRU (Gated Recurrent Unit)
Pérdida vs Época: Similar a LSTM, la pérdida en GRU disminuye de manera rápida durante las primeras 10 épocas, comenzando cerca de 0.7 y alcanzando prácticamente cero al final del entrenamiento. La pérdida se estabiliza en un valor cercano a cero a partir de la época 20, indicando que el modelo está ajustándose de manera eficiente.
Precisión vs Época: En términos de precisión, el modelo GRU muestra una evolución muy destacada. Al igual que LSTM, la precisión aumenta rápidamente desde 0.50 hasta aproximadamente 1.0, y se mantiene casi constante en ese nivel desde la época 20 en adelante. Esto sugiere que el modelo está prácticamente sobreajustando el conjunto de entrenamiento, logrando una precisión casi perfecta.

<div style="display: flex; justify-content: space-around; align-items: center;">

  <div style="text-align: center;">
    <img src="./images/GRU_LOSS.png" alt="Imagen 1" style="width: 600px; height: auto;">
  </div>

  <div style="text-align: center;">
    <img src="./images/GRU_ACC.png" alt="Imagen 2" style="width: 600px; height: auto;">
  </div>

</div>

### ROC AUC
#### 1. RNN (Red Neuronal Recurrente Simple)
Análisis: Los valores AUC para ambas clases están muy cerca del azar (0.5), lo que sugiere que el modelo RNN no está logrando discriminar de manera efectiva entre las clases acusatoria y no acusatoria. La curva ROC se mantiene bastante cerca de la diagonal aleatoria, indicando que el rendimiento del modelo es débil en esta tarea.

<div style="text-align: center;">
  <img src="./images/RNN_ROCAUC.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

#### 2. LSTM (Long Short-Term Memory)
Análisis: LSTM muestra una mejora notable en comparación con RNN, con un AUC de aproximadamente 0.74-0.75 para ambas clases. Esto indica que el modelo es capaz de distinguir mucho mejor entre las clases, y la curva ROC se aleja de la diagonal, evidenciando que el modelo ha capturado patrones útiles en los datos.

<div style="text-align: center;">
  <img src="./images/LSTM_ROCAUC.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

#### 3. GRU (Gated Recurrent Unit)
Análisis: El modelo GRU tiene un rendimiento muy similar al de LSTM, con un AUC de 0.73 para ambas clases. Esto sugiere que, al igual que LSTM, GRU es capaz de captar las dependencias secuenciales en los datos, y es casi igual de efectivo para esta tarea de clasificación.

<div style="text-align: center;">
  <img src="./images/GRU_ROCAUC.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

### Matriz de Consufión
#### 1. RNN (Red Neuronal Recurrente Simple)
Análisis: El modelo RNN presenta un alto número de falsos positivos, clasificando incorrectamente muchas muestras de la clase "No Acusatoria" como "Acusatoria". Solo logra detectar 24 casos verdaderos de "Acusatoria", lo cual es bastante bajo en comparación con los 758 errores. Esto indica que el modelo no está capturando correctamente las diferencias entre las dos clases.

<div style="display: flex; justify-content: space-around; align-items: center;">

  <div style="text-align: center;">
    <img src="./images/RNN_MC.png" alt="Imagen 1" style="width: 600px; height: auto;">
  </div>

  <div style="text-align: center;">
    <img src="./images/RNN_MCP.png" alt="Imagen 2" style="width: 600px; height: auto;">
  </div>

</div>

#### 2. LSTM (Long Short-Term Memory)
Análisis: El modelo LSTM mejora significativamente en la detección de la clase "No Acusatoria", reduciendo el número de falsos positivos (de 758 a 92). Sin embargo, tiene un bajo rendimiento en la identificación de la clase "Acusatoria", detectando correctamente solo 6 casos y fallando en 23 (falsos negativos). A pesar de que es más preciso en "No Acusatoria", su capacidad para identificar casos de "Acusatoria" sigue siendo limitada.

<div style="display: flex; justify-content: space-around; align-items: center;">

  <div style="text-align: center;">
    <img src="./images/LSTM_MC.png" alt="Imagen 1" style="width: 600px; height: auto;">
  </div>

  <div style="text-align: center;">
    <img src="./images/LSTM_MCP.png" alt="Imagen 2" style="width: 600px; height: auto;">
  </div>

</div>

#### 3. GRU (Gated Recurrent Unit)
Análisis: El modelo GRU ofrece un rendimiento muy similar al de LSTM en la detección de la clase "Acusatoria", con solo 6 verdaderos positivos y 23 falsos negativos. Sin embargo, GRU mejora aún más en la detección de la clase "No Acusatoria", con solo 22 falsos positivos, siendo el mejor en este aspecto entre los tres modelos. Esto sugiere que el modelo es bastante confiable en la predicción de "No Acusatoria", aunque su capacidad para identificar "Acusatoria" sigue siendo un desafío.

<div style="display: flex; justify-content: space-around; align-items: center;">

  <div style="text-align: center;">
    <img src="./images/GRU_MC.png" alt="Imagen 1" style="width: 600px; height: auto;">
  </div>

  <div style="text-align: center;">
    <img src="./images/GRU_MCP.png" alt="Imagen 2" style="width: 600px; height: auto;">
  </div>

</div>

## Gráfico TSNE
#### 1. RNN (Red Neuronal Recurrente Simple)
En el gráfico TSNE del modelo RNN, se observa que las clases (0: No Acusatoria y 1: Acusatoria) no están claramente separadas. Las instancias de la clase 1 (Acusatoria) están dispersas entre las de la clase 0 (No Acusatoria), lo que sugiere que el modelo tiene dificultades para encontrar una representación latente que separe correctamente ambas clases. La falta de separación clara puede explicar el bajo rendimiento del modelo en las métricas de precisión y AUC.

<div style="text-align: center;">
  <img src="./images/RNN_TSNE.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

#### 2. LSTM (Long Short-Term Memory)
El gráfico TSNE del modelo LSTM muestra una mejora notable en la agrupación. Las instancias de la clase 0 (No Acusatoria) tienden a agruparse más en diferentes áreas del gráfico, mientras que las instancias de la clase 1 (Acusatoria) siguen dispersas, pero en menor medida. Hay un área clara en la parte superior izquierda del gráfico donde las instancias de la clase 1 están más concentradas, lo que indica que el modelo LSTM es capaz de capturar más patrones en las representaciones latentes, aunque todavía hay un margen para mejorar la separación de clases.

<div style="text-align: center;">
  <img src="./images/LSTM_TSNE.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

#### 3. GRU (Gated Recurrent Unit)
El gráfico TSNE del modelo GRU muestra una estructura de agrupamiento similar a la de LSTM, pero con una ligera mejora en la separación entre las clases. Las instancias de la clase 1 (Acusatoria) están algo mejor agrupadas, en particular en la parte izquierda del gráfico. Aunque todavía hay alguna superposición entre las clases, la separación es más clara en comparación con el modelo RNN. Esto respalda los mejores resultados que hemos visto en las métricas de precisión y AUC para el modelo GRU.

<div style="text-align: center;">
  <img src="./images/GRU_TSNE.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>

## RESULTAODS PREVIOS DE TESIS CON PROMPTING Y LLAMA 3.1 8B
<div style="text-align: center;">
  <img src="./images/TESIS_REPORTE.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>
<div style="text-align: center;">
  <img src="./images/TESIS_ROCAUC.png" alt="Distribución de Frases Acusatorias" style="width: 500px; height: auto;" />
</div>
<div style="text-align: center;">
  <img src="./images/TESIS_MC.png" alt="Distribución de Frases Acusatorias" style="width: 1000px; height: auto;" />
</div>

## CONCLUSIONES
Rendimiento Deficiente de RNN: El modelo RNN mostró el peor desempeño, con una precisión extremadamente baja (0.03) para la clase "Acusatoria" y un F1-score general muy bajo para ambas clases. Esto indica que RNN no es adecuado para esta tarea, ya que tiene dificultades para distinguir entre las clases, con un accuracy total de solo 0.24.

LSTM Mejora la Precisión General: El modelo LSTM presentó una precisión mucho mejor, alcanzando un accuracy de 0.89, pero aún tiene problemas para identificar correctamente la clase "Acusatoria". Con una precisión de solo 0.06 y un F1-score de 0.09 para esta clase, LSTM es mucho más confiable en la predicción de la clase "No Acusatoria" (F1-score de 0.94), pero sigue siendo insuficiente para la clase minoritaria.

GRU Ofrece el Mejor Balance: El modelo GRU logró un rendimiento más equilibrado entre ambas clases, con un accuracy de 0.96. Aunque la clase "Acusatoria" sigue siendo difícil de detectar, GRU mostró una mejora significativa en comparación con LSTM, con una precisión de 0.21 y un F1-score de 0.21 para esta clase. Además, mantuvo un F1-score casi perfecto (0.98) para la clase "No Acusatoria".

Problemas con la Clase Acusatoria: Todos los modelos presentaron dificultades notables en la detección de la clase "Acusatoria". A pesar de las mejoras en precisión y recall con GRU, la baja cantidad de datos en esta clase parece estar afectando su capacidad de generalización, lo que requiere estrategias adicionales como la recolección de más datos o el ajuste de clases desbalanceadas.

GRU es la Mejor Opción Global: En general, el modelo GRU es la mejor opción de los tres para esta tarea. Con una precisión y recall mucho más equilibrados entre ambas clases, y un accuracy general de 0.96, GRU es el modelo más robusto, especialmente en términos de minimizar falsos positivos en la clase "No Acusatoria" y ofrecer una mejor capacidad de detección de la clase "Acusatoria".

## REFERENCIAS
[1] 	Y. Fang, X. Li, S. W. Thomas y X. Zhu, "ChatGPT as Data Augmentation for Compositional Generalization: A Case Study in Open Intent Detection," Proceedings of the Joint Workshop of the 5th Financial Technology and Natural Language Processing (FinNLP) and 2nd Multimodal AI For Financial Forecasting (Muffin), Macao, 2023.

[2] 	H. Dai et al., "AugGPT: Leveraging ChatGPT for Text Data Augmentation," arXiv preprint arXiv:2302.13007v3, Mar. 2023.

[3] 	OECD, "Generative Artificial Intelligence for Anti-Corruption and Integrity in Government," OECD Artificial Intelligence Papers, No. 12, Marzo 2024.

[4] Llama, "Prompting - How-to Guides," Llama Documentation. [Online]. Available: https://www.llama.com/docs/how-to-guides/prompting/. [Accessed: 13-Sep-2024].

[5] F. M. Shiri, T. Perumal, N. Mustapha, and R. Mohamed, "A Comprehensive Overview and Comparative Analysis on Deep Learning Models: CNN, RNN, LSTM, GRU," arXiv preprint, vol. 2305.17473v2, 2023. [Online]. Available: https://arxiv.org/abs/2305.17473v2. [Accessed: 13-Sep-2024].

[6] S. A. Zargar, "Introduction to Sequence Learning Models: RNN, LSTM, GRU," Preprint, North Carolina State University, Apr. 2021. [Online]. Available: https://www.researchgate.net/publication/350950396. [Accessed: 13-Sep-2024].
