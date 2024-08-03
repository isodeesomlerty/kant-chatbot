# Kant Chatbot

This project implements a chatbot that answers questions about Immanuel Kant's philosophy based on his writings.

## Features

- Uses RAG (Retrieval-Augmented Generation) to provide accurate responses
- Implements a Flask web application for easy interaction
- Utilizes OpenAI's language models for natural language processing

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/kant-chatbot.git
   cd kant-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

4. Large files:
   This project requires large data files (`all_embeddings.npy` and `processed_data.pkl`) which are not included in the repository due to size constraints. You need to generate or obtain these files separately and place them in the project root directory.

   The application expects these files to be present for full functionality.

5. Run the application:
   ```
   python app.py
   ```

## Usage

Visit the web interface and start asking questions about Kant's philosophy!

## Deployment

This application is set up for deployment on Heroku. Note that the large data files are not included in the deployment. To deploy this application with full functionality, you'll need to implement a solution to provide the necessary data files in the production environment.

To deploy to Heroku:

1. Create a Heroku app:
   ```
   heroku create your-app-name
   ```

2. Set the OpenAI API key:
   ```
   heroku config:set OPENAI_API_KEY=your_api_key_here
   ```

3. Push to Heroku:
   ```
   git push heroku main
   ```

4. Ensure that you have a method to provide the required data files (`all_embeddings.npy` and `processed_data.pkl`) in your production environment.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
