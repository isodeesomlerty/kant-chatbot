from flask import Flask, render_template, request, jsonify
from chatbot import KantChatbot
import markdown2

app = Flask(__name__)
chatbot = None

def initialize_chatbot():
    global chatbot
    if chatbot is None:
        chatbot = KantChatbot()

@app.route('/')
def home():
    initialize_chatbot()
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    initialize_chatbot()
    user_message = request.json['message']
    response = chatbot.generate_response(user_message)
    html_response = markdown2.markdown(response)
    return jsonify({'response': html_response})

if __name__ == '__main__':
    app.run(debug=True)