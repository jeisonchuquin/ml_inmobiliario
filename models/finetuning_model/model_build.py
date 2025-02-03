import pandas as pd
import numpy as np
import re
import time
import datetime
import torch
import os
import torch.nn as nn
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
pd.options.display.float_format = "{:,.2f}".format



#! Funciones de limpieza de texto
def treat_m2(text: str):
    text = re.sub(r'(m2)|(m²)', ' m²', text)
    return text

def filter_ibans(text: str):
    pattern = r'fr\d{2}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{2}|fr\d{20}|fr[ ]\d{2}[ ]\d{3}[ ]\d{3}[ ]\d{3}[ ]\d{5}'
    text = re.sub(pattern, '', text)
    return text

def remove_space_between_numbers(text: str):
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    return text

def filter_emails(text: str):
    pattern = r'(?:(?!.*?[.]{2})[a-zA-Z0-9](?:[a-zA-Z0-9.+!%-]{1,64}|)|\"[a-zA-Z0-9.+!% -]{1,64}\")@[a-zA-Z0-9][a-zA-Z0-9.-]+(.[a-z]{2,}|.[0-9]{1,})'
    text = re.sub(pattern, '', text)
    return text

def filter_ref(text: str):
    pattern = r'(\(*)(ref|réf)(\.|[ ])\d+(\)*)'
    text = re.sub(pattern, '', text)
    return text

def filter_websites(text: str):
    pattern = r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*'
    text = re.sub(pattern, '', text)
    return text

def filter_phone_numbers(text: str):
    pattern = r'(?:(?:\+|00)33[\s.-]{0,3}(?:\(0\)[\s.-]{0,3})?|0)[1-9](?:(?:[\s.-]?\d{2}){4}|\d{2}(?:[\s.-]?\d{3}){2})|(\d{2}[ ]\d{2}[ ]\d{3}[ ]\d{3})'
    text = re.sub(pattern, '', text)
    return text

def clean_text(text: str):
    text = text.lower()
    text = text.replace(u'\xa0', u' ')
    text = treat_m2(text)
    text = filter_phone_numbers(text)
    text = filter_emails(text)
    text = filter_ibans(text)
    text = filter_ref(text)
    text = filter_websites(text)
    text = remove_space_between_numbers(text)
    return text

def filter_long_descriptions(tokenizer, descriptions, max_len):

    indices = []

    lengths = tokenizer(
        descriptions,
        padding=False,
        truncation=False,
        return_length=True
    )['length']

    for i in range(len(descriptions)):
        if lengths[i] <= max_len-2:
            indices.append(i)

    return indices

def create_dataloaders(inputs, masks, labels, batch_size):

    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor,
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)
    return dataloader

class BetoRegressor(nn.Module):

    def __init__(self, drop_rate=0.2, freeze_beto=False):

        super(BetoRegressor, self).__init__()
        D_in, D_out = 768, 1

        self.beto = BetoModel #.from_pretrained('bert-base-spanish-wwm-cased')
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )
        self.double()

    def forward(self, input_ids, attention_masks):

        outputs = self.beto(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs

def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(model,
        optimizer,
        scheduler,
        loss_function,
        epochs,
        train_dataloader,
        device,
        clip_value=2):

    training_stats = []

    for epoch in range(epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        best_loss = 1e10
        model.train()
        for step, batch in enumerate(train_dataloader):

            if step % 10 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print(f' Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}.')

            # print(step)
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)

            model.zero_grad()

            outputs = model(batch_inputs, batch_masks)

            loss = loss_function(
                outputs.squeeze(),
                batch_labels.squeeze())

            total_train_loss += loss.item()

            loss.backward()

            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print('')
        print(f'Average training loss: {avg_train_loss}')
        print(f'Training epoch took: {training_time}')

        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Training Time': training_time
            }
        )


    return model, training_stats

def r2_score(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def evaluate(model, loss_function, test_dataloader, device):

    model.eval()
    test_loss, test_r2 = [], []

    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)

        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)

        loss = loss_function(outputs, batch_labels)
        test_loss.append(loss.item())
        r2 = r2_score(outputs, batch_labels)
        test_r2.append(r2.item())

    return test_loss, test_r2

def predict(model, dataloader, device):

    model.eval()
    output = []

    for batch in dataloader:
        batch_inputs, batch_masks, _ = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(
                batch_inputs,
                batch_masks
            ).view(1,-1).tolist()[0]

    return output


# seteo de parámetros de conexión a RDS
user = 'postgres'
password = 'password'
host = 'localhost'
port = '5432'
database = 'inmodbcore'
con_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'

engine = create_engine(con_string)
conn = engine.connect()


# extracción de datos, son los mismos que se encuentra en el archivo de excel
sql_query = '''
select
case
	when "TIPO" = 'Derechos y Acciones' then 'Casa'
	else "TIPO"
end as "TIPO"
, "LOCALIZACIONDELBIEN"
, "CARACTERISTICAS"
, "AREAPROPIEDAD"
, "AREACONSTRUCCION"
, "AREATERRENO"
, "AVALUOCALCULADO"
from "REMATES_INTERNO" ri
where "TIPO" not in ('Finca', 'Suite', 'Edificio')
'''

data = pd.read_sql(sql_query, conn)


# eliminamos las que no tienen características ya que esta sería la variable principal
data.dropna(subset=['CARACTERISTICAS'], inplace=True)


data.isnull().sum() / data.shape[0]
# observamos que el AREATERRENO no se puede utilizar debido al % de nulos, sin embargo,
# la podemos utilizar para rellenar las otras columnas de área, pero no voy a utilizarlo


data['AREACONSTRUCCION'] = data.AREACONSTRUCCION.combine_first(data.AREAPROPIEDAD)
data.isnull().sum()


# observamos que no se podría utilizar el área, ya que solo 5 casos tienen la información
data[data.AREACONSTRUCCION.isnull()].isnull().sum()

# por lo que eliminaremos estos datos
data.dropna(subset=['AREACONSTRUCCION'], inplace=True)


# eliminamos las otras columnas que no se utilizaran
cols_delete = [
    'AREAPROPIEDAD',
    'AREATERRENO'
]
data.drop(columns=cols_delete, inplace=True)

data.isnull().sum()


# tenemos una data poblada con buena cantidad de datos
data.shape

# verificamos la distribución de AVALUO
data.AVALUOCALCULADO.describe()


# vemos que hay un valor bastante extremo por lo cual, a pesar de que no se debe hacer,
data.AVALUOCALCULADO.plot(kind='box')
# eliminaremos este valor atípico


# observamos los percentiles, y, el 98% está por debajp de los 6000mil, por lo tanto podriamos eliminar
# los datos mayores a estos, sin embargo, vemos primero cuantos datos son
data.AVALUOCALCULADO.describe(percentiles=[0.8, 0.9, 0.95, 0.98, 0.99])
data.AVALUOCALCULADO.describe(percentiles=[0.1, 0.15, 0.20, 0.25])

data.query('AVALUOCALCULADO >= 600000').shape
data.query('AVALUOCALCULADO >= 344000').shape[0] / data.shape[0]
# son 120 datos que se eliminarian, es decir el 1%


data_filtered = data.query('1000 <= AVALUOCALCULADO <= 344000').copy()
data_filtered.AVALUOCALCULADO.describe()
data_filtered.AVALUOCALCULADO.plot(kind='box')


data_filtered.AVALUOCALCULADO.plot(kind='hist')
# si bien se tiene datos extremos, no obstante los datos están mejores distribuidos

# ahora verifiquemos que la longitud de la descripción es útil
data_filtered.CARACTERISTICAS.apply(len).describe()

data_filtered[data_filtered.CARACTERISTICAS.apply(len) < 50].tail(20)
# la descripción que es menor a 50 caracteres las eliminamos, pues como se puede obervar no dice mucho

data_filtered = data_filtered[data_filtered.CARACTERISTICAS.apply(len) >= 50]
data_filtered.CARACTERISTICAS.apply(len).plot(kind='hist', bins=20)

res = data_filtered.CARACTERISTICAS.str.split(expand=True).stack().value_counts()
res.iloc[: 30].plot(kind='bar')

# observamos que los stopwords son llas palabras más frecuentes, y luego aparencen lo de casas
# como pisos, plantas, áreas, hormigon
# como queremos utilizar toda la información de texto, unimos todas las columnas con la de caracteríasticas,
# dejando de lado la de AVALUO


data_filtered['CUERPO'] = data_filtered.TIPO + \
    ' ' + \
    data_filtered.LOCALIZACIONDELBIEN + \
    ' ' + \
    data_filtered.CARACTERISTICAS + \
    ' ' + \
    data_filtered.AREACONSTRUCCION.astype(str)


data_filtered['CUERPO'] = data_filtered.CUERPO.str.replace('?', '')


#! Preprocesamiento para BERT-BETO
# dividimos la data en train and test
data_filtered.reset_index(drop=True, inplace=True)
train_data, test_data = train_test_split(data_filtered, train_size=0.80, random_state=888)

train_data['cleaned_description'] = train_data.CUERPO.apply(clean_text)

# tokenizamos el texto
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased", do_lower_case=False)
encoded_corpus = tokenizer(text=train_data.cleaned_description.tolist(),
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            max_length=500,
                            return_attention_mask=True)

input_ids = encoded_corpus['input_ids']
attention_mask = encoded_corpus['attention_mask']

# filtramos las características mayores a 300, dependiendo del tiempo que se demore, subiremos este número
short_descriptions = filter_long_descriptions(
    tokenizer,
    train_data.cleaned_description.tolist(),
    500
)

input_ids = np.array(input_ids)[short_descriptions] #contiene las características codificadas de acuerdo a beto
attention_mask = np.array(attention_mask)[short_descriptions]
labels = train_data.AVALUOCALCULADO.to_numpy()[short_descriptions]

# para evaluar el desempeño del entrenamiento, utilizamos el 10% de la data de entreno
test_size = 0.1
seed = 88
train_inputs, test_inputs, train_labels, test_labels = \
            train_test_split(
                input_ids,
                labels,
                test_size=test_size,
                random_state=seed
            )

train_masks, test_masks, _, _ = train_test_split(
    attention_mask,
    labels,
    test_size=test_size,
    random_state=seed
)


# en el proceso de regresión con deep learning, se recomienda scalar la variable objetivo para
# que ayude a la estabilidad del modelo y convergencia
price_scaler = StandardScaler()
price_scaler.fit(train_labels.reshape(-1, 1))

train_labels = price_scaler.transform(train_labels.reshape(-1, 1))
test_labels = price_scaler.transform(test_labels.reshape(-1, 1))

# utilizaremos el framework de pytorch
torch.cuda.get_device_name()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.get_device_properties('cuda')
torch.cuda.empty_cache()

batch_size = 8
train_dataloader = create_dataloaders(
    train_inputs,
    train_masks,
    train_labels,
    batch_size
)

test_dataloader = create_dataloaders(
    test_inputs,
    test_masks,
    test_labels,
    batch_size
)

# definimos nuestro modelo de BETORegressor
# BetoModel = AutoModelForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
BetoModel = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# instanciamos el modelo customizado
model = BetoRegressor(drop_rate=0.2)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

model.to(device)

# definimos la función de pérdida
optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    eps=1e-8
)

# definimos epocas
epochs = 2
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_function = nn.MSELoss()

print(total_steps)

# entrenamiento del modelo
model, training_stats = train(
    model,
    optimizer,
    scheduler,
    loss_function,
    epochs,
    train_dataloader,
    device,
    clip_value=2
)

#! Predicciones
val_data = test_data.copy()

# aplicamos el mismo preprocesamiento al data de entrenamineto
val_set = val_data[['CUERPO', 'AVALUOCALCULADO']]

val_set['cleaned_description'] = val_set.CUERPO.apply(clean_text)

encoded_val_corpus = tokenizer(
    text=val_set.cleaned_description.tolist(),
    add_special_tokens=True,
    padding='max_length',
    truncation='longest_first',
    max_length=300,
    return_attention_mask=True
)

val_input_ids = np.array(encoded_val_corpus['input_ids'])
val_attention_mask = np.array(encoded_val_corpus['attention_mask'])
val_labels = val_set.AVALUOCALCULADO.to_numpy()
val_labels = price_scaler.transform(val_labels.reshape(-1, 1))

val_dataloader = create_dataloaders(
    val_input_ids,
    val_attention_mask,
    val_labels,
    batch_size
)

y_pred_scaled = predict(model, val_dataloader, device)

# devolvemos los precios predichos a la escala original
y_test = val_set.AVALUOCALCULADO.to_numpy()
y_pred = price_scaler.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))

# metricas
mae = mean_absolute_error(y_test, y_pred)
mdae = median_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

pd.DataFrame({'real': y_test, 'prediccion': y_pred.reshape(-1)}, index=range(len(y_test)))
