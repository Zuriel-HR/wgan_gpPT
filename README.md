# wgan_gpPT
WGAN con penalización de gradiente

Este repositorio hace uso de una red neuronal WGAN-GP que toma como datos muestras sintentizadas de malware almacenadas en el archivo: \02_wgan_gp_PT\data\dataset_malware.npy con forma (584, 32, 32, 1).

Se procesan las muestras en el archivo \02_wgan_gp_PT\wgan_gp_train.ipynb y crea dos redes neuronales: crítico y generador para después guardarlas en \02_wgan_gp_PT\models. Este archivo "wgan_gp_train.ipynb" tiene una serie de variables que pueden mejorar o empeorar el entrenamiento como:
1. CRITIC_STEPS: Número de actualizaciones de crítco por actualización de generador.
2. BATCH_SIZE: Lotes en los que se divide la base de datos.
3. Z_DIM: Dimensión del espacio latente.
4. GP_WEIGHT: Coeficiente de regularización que controla cuánto peso tiene el gradient penalty (c_gp) dentro de la pérdida total del crítico.


Después es natural ejecutar los demás archivos:
1. /02_wgan_gp_PT\wgan_gp_generate.ipynb: Genera nuevas muestras, decodifica las muestras de imagen a json para que sean útiles.
2. /02_wgan_gp_PT\wgan_gp_eval.ipynb: Genera una serie de métricas de evaluación. La primera espara destinguir muestras originales con las generadas mediante C2ST AUC.
3. /02_wgan_gp_PT\tensorboard.ipynb: muestra las métricas guardadas en \02_wgan_gp_PT\logs\train, generadas durante el entrenamiento.

