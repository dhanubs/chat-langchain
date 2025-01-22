from openai import OpenAI
import os
from rich import print  # Optional, for prettier output

# Initialize the client with your API key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

try:
    # List all models
    models = client.models.list()
    
    print("\n🔍 Available Models:")
    print("==================")
    
    # Filter and print GPT models
    gpt_models = [model for model in models.data if "gpt" in model.id.lower()]
    for model in gpt_models:
        print(f"✓ {model.id}")
    
    # Specifically check GPT-4 access
    gpt4_models = [model for model in gpt_models if "gpt-4" in model.id.lower()]
    print("\n🔐 GPT-4 Access:")
    print("===============")
    if gpt4_models:
        print("✅ You have access to GPT-4 models:")
        for model in gpt4_models:
            print(f"  - {model.id}")
    else:
        print("❌ No GPT-4 access found")

except Exception as e:
    print(f"❌ Error: {str(e)}") 