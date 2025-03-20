# Chess Agent – AI-Powered Chess Application

*A GUI-based AI Chess Agent that plays against the user with intelligent decision-making.*

## 📌 Overview
The **Chess Agent** is an AI-powered chess application that allows users to play against a smart opponent. Unlike traditional engines, this project utilizes a custom AI model trained using **Minimax with Alpha-Beta Pruning** and **Reinforcement Learning** to evaluate and respond to player moves efficiently.

## 🛠️ Technologies Used
- **Languages & Libraries:** Python, Python-chess, Pandas, Numpy, TensorFlow, Flask
- **AI Techniques:** Minimax Algorithm, Alpha-Beta Pruning, Reinforcement Learning
- **GUI & API:** Flask-based Web Interface

## 🎯 Features
✅ Play against an AI-powered chess agent  
✅ Intelligent decision-making with Minimax and Reinforcement Learning  
✅ Heuristic-based evaluation for optimized move selection  
✅ Fast and efficient search algorithms for real-time gameplay  
✅ Simple and interactive GUI using Flask  

## 🚀 Installation & Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mohak34/ChessBotAI.git
   cd ChessBotAI
   ```
2. **Create and Activate a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application:**
   ```bash
   python app.py
   ```
5. **Open the GUI:**
   - Visit `http://127.0.0.1:5000` in your browser.

## 🎮 How It Works
1. The user makes a move on the GUI.
2. The AI processes the move using **Minimax with Alpha-Beta Pruning**.
3. The AI evaluates possible moves, selects the best option, and plays.
4. The game continues until checkmate or draw.

## 📚 Future Improvements
- Enhance AI efficiency using Deep Reinforcement Learning
- Improve GUI for better user experience
- Train the model with previous data of chess matches to improve the decision making process

## 🤝 Contributing
- Contributions are welcome! Feel free to fork the repo, submit issues, and create pull requests.

