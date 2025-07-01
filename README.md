# TFG - Pokémon Red Vision Model


Final Degree Project for the Computer Engineering program at the University of the Balearic Islands.  
This project is dedicated to developing a segmentation model for Pokémon Red.

This notebook trains a semantic segmentation model using images from the Pokémon video game, labeled via [Roboflow](https://roboflow.com/).

The goal is to enable the model to identify and classify different regions within each image (e.g., walkable paths, obstacles, characters) by performing pixel-wise segmentation.

This vision system is part of the PokemanIA project, which aims to provide game-playing AI agents with a structured perception of their environment. Semantic segmentation enables the creation of a simplified, information-rich map that can support tasks such as navigation, planning, and reinforcement learning.

### Notebook contents:
- Dataset loading and preprocessing  
- Training and evaluation  
- Result visualization  
- Saving the trained model  

---

## 🛠️ Installation & Usage

> **Recommended Python version:** `Python 3.10+`  
> It's highly suggested to use [Anaconda](https://anaconda.org/anaconda/python) for easier environment management.

### Setup Instructions
1. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ````

2. **You're all set!**

To start training, run the training code inside the provided Jupyter Notebook:

- **Run training in Jupyter Notebook:**
Open and execute the cells in train_vision.ipynb to start training your model interactively.

