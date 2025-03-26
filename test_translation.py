import requests
import json

def test_translation():
    url = "http://localhost:8000/test-full-process/"
    data = {
        "text": "Olá, como você está? Estou muito bem, obrigado!"
    }
    
    try:
        response = requests.post(url, json=data)
        print("Status Code:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        print("Erro:", str(e))

if __name__ == "__main__":
    test_translation() 