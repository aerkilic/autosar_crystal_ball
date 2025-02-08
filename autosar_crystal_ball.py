from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
import re
from datasets import Dataset

# Spezialtokens definieren
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[SD]", "[SDCONFIG]", "[LOWERMULTIPLICITY]", "[UPPERMULTIPLICITY]", "[SDSERVERTIMER]", "[SDCLIENTTIMER]", "[SDEVENTHANDLER]", "[SDCONSUMEDEVENTGROUP]", "[SDSERVICEINSTANCE]", "[SDOFFER]", "[SDSUBSCRIBE]", "[SDTXPDUREF]", "[SDRXPDUREF]", "[SERVICEINSTANCEID]", "[SERVICEINTERFACE]", "[SERVICEVERSION]", "[TRANSPORTPROTOCOL]", "[IPCONFIGURATION]", "[DISCOVERYMODE]", "[MANDATORY]"
    ]
}

# JSON-Beispieldaten für AUTOSAR SD mit mehreren Beispielen
json_data = '''{
  "Sd": {
    "SdConfig": {
      "LowerMultiplicity": 1,
      "UpperMultiplicity": "n",
      "SdServiceInstance": [
        {
          "ServiceInstanceID": "Service_1",
          "ServiceInterface": "SomeServiceInterface",
          "ServiceVersion": "1.0.0",
          "TransportProtocol": "SOME/IP",
          "IPConfiguration": {
            "IPAddress": "192.168.1.100",
            "Port": 30509
          },
          "DiscoveryMode": "DynamicDiscovery",
          "Mandatory": true
        },
        {
          "ServiceInstanceID": "Service_2",
          "ServiceInterface": "AnotherServiceInterface",
          "ServiceVersion": "2.1.0",
          "TransportProtocol": "SOME/IP",
          "IPConfiguration": {
            "IPAddress": "192.168.1.101",
            "Port": 30510
          },
          "DiscoveryMode": "OneTime",
          "Mandatory": false
        },
        {
          "ServiceInstanceID": "Service_3",
          "ServiceInterface": "UnknownService",
          "ServiceVersion": "3.0.5",
          "TransportProtocol": "MQTT",
          "IPConfiguration": {
            "IPAddress": "10.0.0.50",
            "Port": 1883
          },
          "DiscoveryMode": "PeriodicDiscovery",
          "Mandatory": true
        },
        {
          "ServiceInstanceID": "Service_4",
          "ServiceInterface": "ExperimentalService",
          "ServiceVersion": "0.9.1",
          "TransportProtocol": "HTTP",
          "IPConfiguration": {
            "IPAddress": "192.168.2.200",
            "Port": 8080
          },
          "DiscoveryMode": "ExperimentalMode",
          "Mandatory": false
        }
      ]
    }
  }
}'''

# JSON laden
data = json.loads(json_data)

# BERT Tokenizer laden und Spezialtokens hinzufügen
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens(SPECIAL_TOKENS)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.resize_token_embeddings(len(tokenizer))

# Funktion zum Umwandeln von JSON-Schlüsseln in Spezialtokens
def convert_json_to_special_tokens(data):
    text = ""
    if "LowerMultiplicity" in data:
        text += f"[LOWERMULTIPLICITY] {data['LowerMultiplicity']} "
    if "UpperMultiplicity" in data:
        text += f"[UPPERMULTIPLICITY] {data['UpperMultiplicity']} "
    return text.strip()

# Dataset vorbereiten
def prepare_dataset(data):
    samples = []
    for instance in data["Sd"]["SdConfig"]["SdServiceInstance"]:
        mandatory_token = "[MANDATORY]" if instance["Mandatory"] else "[OPTIONAL]"
        instance_tokens = " ".join([f"[{key.upper()}] {value}" for key, value in instance.items() if key != "Mandatory"])
        service_info = convert_json_to_special_tokens(data["Sd"]["SdConfig"])
        text = f"{mandatory_token} {instance_tokens} {service_info}"
        label = 1 if instance["Mandatory"] else 0
        samples.append({"text": text, "label": label})
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

# --- Vorhersage (Prediction) ---

def predict(text):
    instance = json.loads(text)
    mandatory_token = "[MANDATORY]" if instance["Mandatory"] else "[OPTIONAL]"
    instance_tokens = " ".join([f"[{key.upper()}] {value}" for key, value in instance.items() if key != "Mandatory"])
    service_info = convert_json_to_special_tokens(data["Sd"]["SdConfig"])
    text = f"{mandatory_token} {instance_tokens} {service_info}"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return "Mandatory" if predicted_class == 1 else "Optional"

# Testfälle mit vorhergesehenen und unvorhergesehenen Daten
for instance in data["Sd"]["SdConfig"]["SdServiceInstance"]:
    print(f"Prediction for {instance['ServiceInstanceID']}:", predict(json.dumps(instance)))

# Unvorhergesehenes Beispiel testen
unexpected_service = {
    "ServiceInstanceID": "Service_Unknown",
    "ServiceInterface": "NewExperimentalInterface",
    "ServiceVersion": "5.2.0",
    "TransportProtocol": "WebSockets",
    "IPConfiguration": {
        "IPAddress": "10.1.1.50",
        "Port": 9090
    },
    "DiscoveryMode": "UnknownMode",
    "Mandatory": False
}
print("Prediction for unexpected service:", predict(json.dumps(unexpected_service)))
