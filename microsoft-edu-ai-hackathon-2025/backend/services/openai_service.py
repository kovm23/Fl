import os
import openai
import numpy as np
from PIL import Image
import base64
import io
from dotenv import load_dotenv
import json
import time

load_dotenv()

# --- Konfigurace klientů ---

# 1. Azure Client
azure_client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# 2. Local Client (Ollama)
# Předpokládá se běh na defaultním portu 11434
local_client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
)

def image_to_base64(img_arr):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def _get_client_and_model(model_selection):
    """
    Rozhodne, zda použít Azure nebo Local Ollama.
    Pokud model_selection je None nebo obsahuje 'gpt', jde na Azure.
    Jinak jde na localhost.
    """
    if not model_selection or "gpt" in model_selection.lower():
        deployment = os.getenv("AZURE_OPENAI_GPT41_DEPLOYMENT_NAME")
        return azure_client, deployment
    else:
        # Vrátí lokálního klienta a název modelu (např. 'llava:34b')
        return local_client, model_selection

def _clean_json_response(content):
    """Pomocná funkce pro vyčištění Markdownu z JSON odpovědi (časté u lokálních modelů)."""
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[0]
    return content.strip()

def extract_image_features_with_llm(image_base64_list, prompt=None, deployment_name=None, feature_gen=False) -> list:
    features_list = []
    
    # Výběr klienta a modelu
    client, model_name = _get_client_and_model(deployment_name)

    for img_b64 in image_base64_list:
        if prompt is None:
            prompt_text = "Extract meaningful features from this image for tabular dataset construction."
        else:
            prompt_text = prompt
            
        user_content = [{"type": "text", "text": prompt_text}]
        
        # Přidání obrázku (formát je kompatibilní pro GPT-4o i Ollama)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
        
        max_retries = 3
        backoff = 2
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a feature extraction assistant. You MUST output valid JSON only. No text, no markdown, just JSON."},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=2048,
                    temperature=0.1 # Nízká teplota pro stabilnější JSON
                )
                content = response.choices[0].message.content
                clean_content = _clean_json_response(content)
                
                try:
                    features = json.loads(clean_content)
                except Exception:
                    features = {"features": clean_content, "error": "JSON parse error", "raw": content}
                
                features_list.append(features)
                break
            
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    features_list.append({"error": "Rate limit exceeded."})
            except Exception as e:
                features_list.append({"error": f"Model error ({model_name}): {str(e)}"})
                break
                
    return features_list

def extract_text_features_with_llm(text_list, prompt=None, deployment_name=None, feature_gen=False) -> list:
    features_list = []
    client, model_name = _get_client_and_model(deployment_name)
    
    for text in text_list:
        if prompt is None:
            prompt_text = "Extract meaningful features from this text."
        else:
            prompt_text = prompt
            
        # Úprava promptu pro JSON enforcement
        system_prompt = prompt_text
        if feature_gen:
             system_prompt += "\nIMPORTANT: Return ONLY valid JSON."

        max_retries = 3
        backoff = 2
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=2048,
                    temperature=0.1
                )
                content = response.choices[0].message.content
                clean_content = _clean_json_response(content)
                
                try:
                    features = json.loads(clean_content)
                except Exception:
                    features = {"features": clean_content}
                features_list.append(features)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                else:
                    features_list.append({"error": str(e)})
                    break
                    
    return features_list