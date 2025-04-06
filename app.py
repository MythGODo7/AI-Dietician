from flask import Flask, render_template, request
import requests
import re
import logging
import time

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Hugging Face API details
API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Retry logic
def query_huggingface(payload, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    return {"error": f"Failed after {retries} attempts."}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        form_data = request.form.to_dict()
        logger.debug(f"Form data received: {form_data}")

        age = form_data.get('age', '')
        gender = form_data.get('gender', '')
        weight = form_data.get('weight', '')
        height = form_data.get('height', '')
        veg_or_nonveg = form_data.get('veg_or_nonveg', '')
        disease = form_data.get('disease', '')
        region = form_data.get('region', 'Unknown Region')
        allergics = form_data.get('allergics', '')
        foodtype = form_data.get('foodtype', '')

        if not all([age, gender, weight, height, veg_or_nonveg, region]):
            return "❌ Please fill in all required fields."

        prompt = (
            f"You are a dietician and fitness coach. Based on the following data, provide strictly formatted recommendations:\n"
            f"Age: {age}\nGender: {gender}\nWeight: {weight}kg\nHeight: {height}m\nDiet: {veg_or_nonveg}\n"
            f"Disease: {disease or 'None'}\nRegion: {region}\nAllergics: {allergics or 'None'}\nFoodtype: {foodtype or 'Any'}\n\n"
            f"Format:\n"
            f"Restaurants:\n1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n6. ...\n"
            f"Breakfast:\n1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n6. ...\n"
            f"Dinner:\n1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            f"Workouts:\n1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n6. ..."
        )

        hf_response = query_huggingface({"inputs": prompt})
        if "error" in hf_response:
            return f"❌ An error occurred: {hf_response['error']}"

        generated_text = hf_response[0].get("generated_text", "") if isinstance(hf_response, list) else hf_response.get("generated_text", "")
        logger.debug(f"Generated text: {generated_text}")

        def extract_list(section):
            pattern = rf"{section}:\n(.*?)\n(?:\w+:|$)"
            match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
            return [item.strip()[3:].strip() for item in match.group(1).split('\n') if item.strip()] if match else []

        return render_template(
            'result.html',
            restaurant_names=extract_list("Restaurants")[:6],
            breakfast_names=extract_list("Breakfast")[:6],
            dinner_names=extract_list("Dinner")[:5],
            workout_names=extract_list("Workouts")[:6],
            region=region
        )

    except Exception as e:
        logger.error(f"Exception in recommend route: {e}")
        return f"❌ Internal error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
