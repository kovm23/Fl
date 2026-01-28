import os
import openai
import requests
import numpy as np
from PIL import Image
import base64
import io
from dotenv import load_dotenv
import json
import time

load_dotenv()

# --- KONFIGURACE MODELŮ ---
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
MISTRAL_MODEL = "pixtral-12b-2409"

def get_llm_client(provider: str):
    # 1. Azure OpenAI
    if provider == "azure":
        return openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    # 2. OpenRouter (Google Gemini)
    elif provider == "openrouter":
        return {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "model": OPENROUTER_MODEL, 
        }
    # 3. Mistral AI (Pixtral)
    elif provider == "mistral":
        return {
            "base_url": "https://api.mistral.ai/v1",
            "api_key": os.getenv("MISTRAL_API_KEY"),
            "model": MISTRAL_MODEL, 
        }
    # 4. Local Qwen (Ollama)
    elif provider == "ollama_qwen":
        return {
            "base_url": "http://ollama:11434/v1",
            "api_key": "ollama",
            "model": "qwen2.5-vl",
        }
    # 5. Local Llama (Ollama)
    elif provider == "ollama_llama":
        return {
            "base_url": "http://ollama:11434/v1",
            "api_key": "ollama",
            "model": "llama3.2-vision",
        }
    else:
        # Fallback
        return openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )


def image_to_base64(img_arr):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_image_features_with_llm(
    image_base64_list,
    prompt=None,
    deployment_name=None,
    feature_gen=False,
    provider="azure",
):
    features_list = []
    for img_b64 in image_base64_list:
        prompt_text = (
            prompt
            or "Extract meaningful features from this image for tabular dataset construction. Return raw JSON only, no markdown formatting."
        )

        # --- A) LOGIKA PRO AZURE ---
        if provider == "azure":
            azure_client = get_llm_client("azure")
            user_content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ]
            max_retries = 3
            backoff = 2
            for attempt in range(max_retries):
                try:
                    response = azure_client.chat.completions.create(
                        model=deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                        messages=[
                            {"role": "system", "content": "You are a feature extraction assistant. Return ONLY JSON."},
                            {"role": "user", "content": user_content},
                        ],
                        max_tokens=2048,
                        temperature=0.0,
                    )
                    content = response.choices[0].message.content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    features = json.loads(content)
                    features_list.append(features)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                         features_list.append({"error": str(e)})
                    time.sleep(backoff)
                    backoff *= 2

        # --- B) UNIVERZÁLNÍ LOGIKA (OpenRouter, Mistral, Ollama) ---
        elif provider in ["ollama_qwen", "ollama_llama", "openrouter", "mistral"]:
            client = get_llm_client(provider)
            
            url = f"{client['base_url']}/chat/completions"
            headers = {
                "Authorization": f"Bearer {client['api_key']}",
                "Content-Type": "application/json"
            }
            # OpenRouter headers
            if provider == "openrouter":
                headers["HTTP-Referer"] = "http://localhost:5173"
                headers["X-Title"] = "Media Feature Lab"

            payload = {
                "model": client["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.0,
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                if response.status_code == 200:
                    resp_json = response.json()
                    content = resp_json['choices'][0]['message']['content']
                    try:
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()
                        features = json.loads(content)
                    except:
                        features = {"raw_output": content}
                    features_list.append(features)
                else:
                    features_list.append(
                        {"error": f"HTTP {response.status_code}: {response.text}"}
                    )
            except Exception as e:
                features_list.append({"error": str(e)})
                
    return features_list


def extract_text_features_with_llm(
    text_list, prompt=None, deployment_name=None, feature_gen=False, provider="azure"
) -> list:
    features_list = []
    for text in text_list:
        if prompt is None:
            prompt_text = "Extract meaningful features from this text. Return JSON."
        else:
            prompt_text = prompt
            
        if provider == "azure":
            azure_client = get_llm_client("azure")
            try:
                response = azure_client.chat.completions.create(
                    model=deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that returns only JSON."},
                        {"role": "user", "content": f"{prompt_text}\n\nTEXT:\n{text}"}
                    ],
                    max_tokens=2048,
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                try:
                    features = json.loads(content)
                except:
                    features = {"features": content}
                features_list.append(features)
            except Exception as e:
                features_list.append({"error": str(e)})

        elif provider in ["ollama_qwen", "ollama_llama", "openrouter", "mistral"]:
            client = get_llm_client(provider)
            url = f"{client['base_url']}/chat/completions"
            headers = {
                "Authorization": f"Bearer {client['api_key']}",
                "Content-Type": "application/json"
            }
            if provider == "openrouter":
                headers["HTTP-Referer"] = "http://localhost:5173"
                headers["X-Title"] = "Media Feature Lab"

            payload = {
                "model": client["model"],
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that returns only JSON."},
                    {"role": "user", "content": f"{prompt_text}\n\nTEXT:\n{text}"}
                ],
                "max_tokens": 2048,
                "temperature": 0.0,
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    resp_json = response.json()
                    content = resp_json['choices'][0]['message']['content']
                    try:
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()
                        features = json.loads(content)
                    except:
                        features = {"raw_output": content}
                    features_list.append(features)
                else:
                    features_list.append({"error": f"HTTP {response.status_code}: {response.text}"})
            except Exception as e:
                features_list.append({"error": str(e)})
    return features_list