from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
import re
import os
from datasets import Dataset

# JSON-Beispieldaten für AUTOSAR SD mit mehreren Beispielen
json_data = '''{
    "SdGeneral": {
      "SdDevErrorDetect": {
        "value": "true",
        "mandatory": "true"
      },
      "mandatory": "true"
    }
}'''




# JSON laden
data = json.loads(json_data)

# **Alle Schlüsselwörter extrahieren und als Spezialtokens definieren**
def extract_keys(obj, tokens=None):
    if tokens is None:
        tokens = set()  # Set nur einmal initialisieren

    if isinstance(obj, dict):  # Falls obj ein Dictionary ist, durchlaufe seine Schlüssel
        for key, value in obj.items():
            tokens.add(f"[{key.upper()}]")  # Schlüssel in Spezialtoken umwandeln
            extract_keys(value, tokens)  # Rekursiver Aufruf für die Werte

    elif isinstance(obj, list):  # Falls obj eine Liste ist, durchlaufe die Elemente
        for item in obj:
            extract_keys(item, tokens)  # Rekursiver Aufruf für jedes Listenelement

    return tokens  # Das vollständige Token-Set zurückgeben


# Spezialtokens extrahieren und in ein Dictionary packen
special_tokens = {"additional_special_tokens": list(extract_keys(data))}

# BERT Tokenizer laden und Spezialtokens hinzufügen
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens(special_tokens)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.resize_token_embeddings(len(tokenizer))

# **Funktion zur Konvertierung des JSON in Spezialtoken-Text**
def convert_json_to_special_tokens(obj):
    if isinstance(obj, dict):
        text = " ".join([f"[{key.upper()}] {convert_json_to_special_tokens(value)}" for key, value in obj.items()])
    elif isinstance(obj, list):
        text = " ".join([convert_json_to_special_tokens(item) for item in obj])
    else:
        text = str(obj)
    return text

# **Dataset vorbereiten – jetzt mit echten Labels!**
# **Dataset vorbereiten – rekursiv für alle Parameter mit "mandatory"**
def prepare_dataset(data):
    samples = []

    def traverse(obj, parent_key=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}.{key}" if parent_key else key  # Hierarchie beibehalten
                
                if isinstance(value, dict):
                    traverse(value, new_key)  # Rekursiv weitermachen
                    
                    # Falls dieses Dict ein "mandatory"-Attribut besitzt, als Sample hinzufügen
                 #   if "mandatory" in value:
                 #       text = f"[{new_key.upper()}] {convert_json_to_special_tokens(value)}"
                 #       label = 1 if value["mandatory"] == "true" else 0
                 #       samples.append({"text": text, "label": label})
                
                elif isinstance(value, list):
                    for item in value:
                        traverse(item, new_key)  # Falls es eine Liste ist, jeden Eintrag prüfen

                elif isinstance(value, str):
                  if "mandatory" in key:
                    text = f"[{new_key.upper()}] {convert_json_to_special_tokens(value)}"
                    label = 1 if value == "true" else 0
                    samples.append({"text": text, "label": label})
    traverse(data)
    return samples

dataset = Dataset.from_list(prepare_dataset(data))
dataset = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)

# GPU-Unterstützung aktivieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Trainingsparameter
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Training starten
trainer.train()
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Trainiertes Modell laden
model = BertForSequenceClassification.from_pretrained("./trained_model").to(device)
tokenizer = BertTokenizer.from_pretrained("./trained_model")
model.eval()

# **Vorhersage-Funktion**
def predict(json_text):
    instance = json.loads(json_text)  # JSON laden

    predictions = {}  # Dictionary zur Speicherung der Vorhersagen

    def traverse_and_predict(obj, parent_key=""):
        # 1) Wenn 'obj' ein Dictionary ist
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}.{key}" if parent_key else key

                # 2) Wenn 'value' selbst ein Dictionary oder eine Liste ist, weiter rekursiv
                if isinstance(value, dict):
                    traverse_and_predict(value, new_key)
                elif isinstance(value, list):
                    for item in value:
                        traverse_and_predict(item, new_key)

                # 3) Wenn 'value' ein String ist UND key == "mandatory" -> klassifizieren
                elif isinstance(value, str) and key == "mandatory":
                    # Spezial-Token-Text aufbereiten
                    text_for_inference = f"[{new_key.upper()}] {convert_json_to_special_tokens(value)}"
                    
                    inputs = tokenizer(
                        text_for_inference,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True
                    )
                    # Auf GPU verschieben (falls vorhanden)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Kein Gradienten-Tracking für Inferenz
                    with torch.no_grad():
                        outputs = model(**inputs)

                    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
                    # Klassenergebnis ins Dictionary eintragen
                    predictions[new_key] = "Mandatory" if predicted_class == 1 else "Optional"

        # 4) Wenn 'obj' eine Liste ist, rekursiv durchlaufen
        elif isinstance(obj, list):
            for item in obj:
                traverse_and_predict(item, parent_key)

    # Rekursiven Durchlauf starten
    traverse_and_predict(instance)

    return predictions




#print("Prediction for unexpected service:", predict(json.dumps(unexpected_service)))
print("Prediction for unexpected service:", predict(json_data))
