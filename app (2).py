from flask import Flask, request, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can replace this with your custom model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']  # Get the user's input
    inputs = tokenizer.encode(prompt, return_tensors='pt')  # Tokenize the input
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output
    return render_template('index.html', prompt=prompt, result=result)

if __name__ == '__main__':
    app.run(debug=True)
