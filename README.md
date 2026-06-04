# 🐜 Ant Colony Algorithm for Word Sense Disambiguation

---

## 📌 What this project is about

This project solves the problem of Word Sense Disambiguation (WSD).

That means:

> choosing the correct meaning of a word depending on its context.

### Example:

“bank” → financial institution OR river bank

Instead of using machine learning, this project uses a biologically inspired algorithm based on ant colonies.

---

## 🧠 Main idea (simple)

We build a graph where:

- words from sentences are nodes  
- possible meanings (WordNet senses) are also nodes  

Then we simulate ants moving through this graph to find the best meaning for each word.

---

## 🧩 Graph structure

For each ambiguous word:

- a word node  
- multiple sense nodes (called nests)

So basically:

> one word → multiple possible meanings

---

## 🐜 What ants do

Each ant is a simple agent that:

- starts from a sense (nest)  
- moves through the graph  
- prefers “good” paths  
- collects energy when it finds useful context  
- leaves pheromones on paths it travels  

---

## 🔁 How the process works (cycles)

The algorithm runs for multiple cycles.

In each cycle:

- new ants can be created  
- ants move through the graph  
- they collect energy  
- they leave pheromones on edges  
- old pheromones slowly evaporate  
- some ants die when their lifespan ends  

---

## 🎯 Final decision

At the end, for each word:

- we look at all possible senses  
- we pick the best one using:
  - energy-based scoring (best performance)
  - or majority vote (less stable)

---

## 📊 Results

On the SemEval 2007 dataset:

- Energy-based selection: ~0.71 F1  
- Majority vote: ~0.36 F1  

---

## ⚙️ Why it works

The intuition is:

- good senses attract more ants  
- more ants → more pheromones  
- pheromones attract even more ants  
- bad paths slowly disappear (evaporation)

So the system gradually converges to the correct meanings.

---

## 🚀 How to run

```bash
python main.py
```

## 🧪 Main components

- Graph → words + possible senses  
- Ants → agents that explore  
- Energy → how good a path/sense is  
- Pheromones → memory of good decisions  
- Lesk measure → semantic similarity between meanings  

---

## 💡 Big picture

In short:

> ants collectively “vote” for the correct meaning of each word through movement, energy, and pheromones.
