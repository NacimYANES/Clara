import pandas as pd
import time
import openai
import os
from dotenv import load_dotenv
load_dotenv()
# Load dataset
df = pd.read_csv("MIMIC_III_finale.csv")
df = df.head(10000)

# Initialize OpenAI client
client = openai.OpenAI(api_key="OPENAI_API_KEY")

def extract_medical_info(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract 'Maladie chronique', 'symptômes', 'allergies' and 'traitement régulier' from the following medical text."},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

def parse_medical_info(parts):
    chronic, sympt, allergies, treatment = "", "", "", ""
    current_section = None
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if "Maladie chronique" in part:
            current_section = "chronic"
            if part == "- Maladie chronique:":
                continue
            chronic = part.replace("- Maladie chronique:", "").strip()
        elif "Symptômes" in part:
            current_section = "sympt"
            if part == "- Symptômes:":
                continue
            sympt = part.replace("- Symptômes:", "").strip()
        elif "Allergies" in part:
            current_section = "allergies"
            if part == "- Allergies:":
                continue
            allergies = part.replace("- Allergies:", "").strip()
        elif "Traitement régulier" in part:
            current_section = "treatment"
            if part == "- Traitement régulier:":
                continue
            treatment = part.replace("- Traitement régulier:", "").strip()
        elif current_section == "chronic":
            chronic = part
        elif current_section == "sympt":
            sympt = part
        elif current_section == "allergies":
            allergies = part
        elif current_section == "treatment":
            treatment = part
            
    return chronic, sympt, allergies, treatment

# Initialize columns for extracted data
df["Maladie_chronique"] = ""
df["Symptômes"] = ""
df["Allergies"] = ""
df["Traitement_régulier"] = ""

# Process in batches of 10
batch_size = 10
total_rows = len(df)

for start_idx in range(0, total_rows, batch_size):
    end_idx = min(start_idx + batch_size, total_rows)
    print(f"Processing rows {start_idx} to {end_idx-1}...")
    
    # Process batch
    for index in range(start_idx, end_idx):
        text = df.iloc[index]["TEXT"]
        extracted_info = extract_medical_info(text)
        
        if extracted_info:
            parts = extracted_info.split("\n")
            print(f"Processing row {index}:")
            chronic, sympt, allergies, treatment = parse_medical_info(parts)
            
            # Debug print
            print(f"Extracted Maladie chronique: {chronic[:50]}...")
            print(f"Extracted Symptômes: {sympt[:50]}...")
            print(f"Extracted Allergies: {allergies[:50]}...")
            print(f"Extracted Traitement régulier: {treatment[:50]}...")
            print("-" * 50)
            
            df.at[index, "Maladie_chronique"] = chronic
            df.at[index, "Symptômes"] = sympt
            df.at[index, "Allergies"] = allergies
            df.at[index, "Traitement_régulier"] = treatment
        
        time.sleep(1)  # Rate limiting
    
    # Save after each batch
    temp_filename = f"mimic3_dataset_updated_batch_{start_idx//batch_size}.csv"
    df.to_csv(temp_filename, index=False)
    print(f"Batch {start_idx//batch_size} completed and saved to {temp_filename}")

# Save final complete dataset
df.to_csv("mimic3_dataset_updated_final.csv", index=False)
print("Processing complete. Final dataset saved to mimic3_dataset_updated_final.csv")
