# LLM-Powered Support System Prototype

This repository contains the code, data, and notebooks for a research project exploring **fine-tuning and evaluation of large language models (LLMs)** in the context of an **automated support ticket management system**.  
The work combines **data preprocessing, retriever components, fine-tuning pipelines, validation frameworks, and orchestration scripts** to build a prototype support system powered by LLMs.

---

## Repository Structure

### Environment & Dependencies
- **`llm_environment.yml`** – Conda environment specification.  
- **`requirements.txt`** – Python package requirements.

### Data & Preprocessing
- **`Data-llm-transform.py`**, **`data-llm-transform.txt`** – Scripts and resources for text transformation and preprocessing.  
- **`Data-llm-summarize.py`**, **`data-llm-summarize.txt`** – Scripts and resources for summarization preprocessing.  
- **`Retriever-training.py`**, **`retriever-training.txt`** – Scripts and configuration for retriever training.  
- **`Data.ipynb`**, **`SageParser.ipynb`** – Interactive notebooks for data exploration and parsing.  
- **`Data/`** – Dataset storage.  

### Retrieval Components
- **`VectorDB.py`**, **`VectorDB.ipynb`** – Vector database integration (e.g., FAISS).  
- **`DualBERTRetriever.py`** – Dual BERT retriever implementation.  
- **`Retriever.ipynb`** – Retriever experiments and evaluations.  
- **`FAISS/`** – FAISS index storage.

### Generator / Fine-Tuning
- **`GeneratorDataClasses.py`** – Data classes for generator tasks.
- **`LLMClasses.py`** – Core LLM classes containing custom modules and a model management utility.  
- **`Generator-CPT.py`**, **`generator-cpt.txt`** – Generator fine-tuning with a CPT setup.  
- **`Generator-SFT.py`**, **`generator-sft.txt`** – Generator fine-tuning with an SFT setup.  
- **`Generator.ipynb`** – Generator experiments and evaluations.  
- **`LLaMa/`** – Directory for LLaMA-related experiments.  
- **`SBERT/`** – Directory for SBERT-related experiments.  

### Validation Framework
Validation was designed around using LLMs as judges of model outputs:
- **`Validation-original.py`**, **`validation-original.txt`** – Baseline validation.  
- **`Validation-CPT.py`**, **`validation-cpt.txt`** – Validation for CPT fine-tuning.  
- **`Validation-SFT.py`**, **`validation-sft.txt`** – Validation for SFT fine-tuning.  
- **`Validation-LLM-as-a-Judge-Or.py`**, **`validation_llm_as-a-judge-or.txt`** – Original LLM-as-a-Judge evaluation.  
- **`Validation-LLM-as-a-Judge-SFT.py`**, **`validation_llm_as-a-judge-sft.txt`** – Judge-based validation for SFT fine-tuning.  
- **`validation_llm_as-a-judge.txt`** – General LLM-as-a-Judge validation results.  
- **`Validation.ipynb`** – Interactive validation workflow.  

### Application Layer
- **`Application.py`** – Orchestration script gluing together retriever, generator components via a chat application. 
---

## Project Focus

This prototype project explored:
- **Data preprocessing pipelines** for support tickets.  
- **Retriever training** (DualBERT, FAISS vector DB).  
- **Fine-tuning strategies** (SFT, CPT) on domain-specific support data.  
- **Validation frameworks**, including the use of **LLMs as judges** to evaluate generated outputs.  
- **Application orchestration** to integrate components into a single prototype support system.  

The research focus was on **fine-tuning LLMs and validating results**, while the engineering focus was on **building a modular prototype that could integrate with a support ticket management system**.

---

## ⚡ Getting Started

1. **Set up environment**
   ```bash
     conda env create -f llm_environment.yml
     conda activate llm-support
   ```
or install via:
   ```bash
     pip install -r requirements.txt
   ```

2. Run notebooks for data preprocessing, retriever training, or generator fine-tuning (Data.ipynb, Retriever.ipynb, Generator.ipynb).

3. Validate models using Validation.ipynb or Python scripts (Validation-*.py).

4. Run application prototype:
  ```bash
    python Application.py
  ```

---

## Notes

The repository is structured for research and experimentation rather than production deployment.

Some directories (e.g., Data/, LLaMa/, SBERT/, FAISS/) are expected to contain pre-downloaded models or datasets and are not included here. Data contains private informate, hence it cannot be supplied within a public repository.

Implementation details are outlined in the extended researh report.

