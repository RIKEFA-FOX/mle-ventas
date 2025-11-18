# Proyecto: Predecir Ventas Telefono Celular

Airlines es una empresa de transporte aéreo que recientemente realizó una alianza con
una empresa de telecomunicaciones para la venta de equipos de comunicación.
Elabora un modelo predictivo que cubra las necesidades de la aerolínea, para potenciar la
marca y realizar una venta más efectiva de equipos de comunicacion.

# Estructura
```
mle_ventas/
├─ src/
│  ├─ make_dataset.py
│  ├─ train_model.py
│  └─ predict_model.py
├─ notebooks/
│  ├─ N2_BuildingScripts.ipynb
│  └─ N1_model_project_aerolinea.ipynb
├─ data/
├─ models/
├─ requirements.txt
├─ config.yaml
└─ .gitignore
```
# Pasos para la ejecución
## Paso 1: Clonar el Proyecto
```
git clone https://github.com/<USER>/mle-ventas.git
```
## Paso 2: Instalar los pre-requisitos
```
cd mle-ventas/

pip install -r requirements.txt
```
## Paso 3: Ejecutar las pruebas en el entorno
```
python src/make_dataset.py
python src/train_model.py
python src/predict_model.py
```
## Paso 4: Guardar los cambios en el Repo
```
git add .

git commit -m "Pruebas Finalizadas"

git push
```
