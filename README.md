📦 Projeto: Compressão de Modelos TinyML com Foco em Eficiência Energética

Este projeto é parte da dissertação de mestrado de Clariele de Almeida Pereira e investiga o impacto de diferentes técnicas de compressão de modelos TinyML — especificamente tinyCNN, tinyMLP e tinyKNN — sobre métricas de desempenho, consumo de energia e uso de memória, com foco em aplicações embarcadas em dispositivos com recursos computacionais limitados.

Três abordagens de compressão de modelos são avaliadas utilizando a biblioteca PyTorch:

Poda (Pruning)

Quantização (Quantization)

Knowledge Distillation (KD)

A primeira etapa do estudo concentra-se na poda, comparando:

Poda estruturada vs. poda não estruturada

Dois critérios de seleção dos elementos a serem removidos:

Poda randômica

Poda por magnitude (utilizando norma L1)

🧠 Objetivo

Avaliar e comparar os efeitos das diferentes técnicas de compressão aplicadas a modelos TinyML utilizados em uma tarefa de regressão de precipitação, levando em consideração os seguintes aspectos:

🔹 Erro preditivo: MAE, RMSE, R²

🔹 Consumo de energia:

Energia total (J)

Energia por inferência (µJ)

🔹 Tempo de inferência (total e por amostra)

🔹 Uso de memória RAM

🔹 Potência média e de pico

🔹 Corrente e tensão médias durante a inferência

🔧 Tecnologias e Ferramentas

Python 3.x

PyTorch

pandas, numpy, matplotlib

Medidor de energia baseado em INA219 para medições físicas (em dispositivos embarcados como Raspberry Pi)

Scripts compatíveis com execução em CPU e GPU (com ou sem medição real de energia)

📊 Resultados

Cada execução gera um arquivo .csv contendo as seguintes métricas:

MAE, RMSE, R²

Energia total (J) e energia por inferência (µJ)

Tempo total e tempo por amostra (s)

Potência média e de pico (W)

Corrente e tensão médias (A, V)

Uso de memória antes e depois da inferência

Taxa de esparsidade aplicada

👩‍🔬 Autoria

Clariele de Almeida Pereira
Mestranda em Ciência da Computação — UFRPE
Projeto orientado por [Prof. Dr. Ermeson Andrade e Prof. Dr. Danilo Araújo]
