# Official Repository of EMNLP 2024 (Main) Paper  
**“Quis custodiet ipsos custodes?” Who Will Watch the Watchmen? On Detecting AI-Generated Peer Reviews**

This is official Repository of EMNLP 2024 (Main) Paper `Quis custodiet ipsos custodes?' Who will watch the watchmen? On Detecting AI-generated Peer Reviews

## 🚨 Introduction  
What if peer reviews themselves are AI-generated? Who will guard the guards?

The integrity of the peer-review process is vital for maintaining scientific rigor and trust within the academic community. With the increasing use of large language models (LLMs) like ChatGPT in academic writing, concerns about AI-generated texts compromising scientific publishing, including peer reviews, are growing.

While previous work has focused on generic AI-generated text detection or estimating the fraction of AI-generated peer reviews, this paper addresses a critical real-world problem: **helping editors or chairs determine whether a review is authored by ChatGPT.**

---



## 💡 Key Contributions  
1. **Term Frequency (TF) Model**  
   - Based on the hypothesis that AI-generated texts frequently repeat tokens.

2. **Review Regeneration (RR) Model**  
   - Relies on the principle that ChatGPT generates similar outputs when re-prompted.

3. **Stress Testing Against Attacks**  
   - Models are evaluated against token attacks and paraphrasing.

4. **Defensive Strategy**  
   - Proposes methods to reduce the impact of paraphrasing, improving model robustness.

Our findings suggest that both models outperform existing AI text detectors, with the **RR model** demonstrating greater robustness under attack conditions. The **TF model** performs better in the absence of attacks.

---

## 🚀 Try It Yourself  
Explore our models on Hugging Face:  
[Hugging Face Demo](https://huggingface.co/spaces/AnonymousBabu/frequency-based-ai-text-detection)

---

## Repository Structure:- 
- `Dataset` : This contains all the datasets in JSON format that have been used in the research. In each JSON file, keys are the IDs of the open review paper, and the values are the reviews of it.
- `Embeddings`: This contains the embeddings of the dataset created from the open-ai text-embedding-3-small model
- `Dictionaries`: This contains the dictionaries that have been used in our token-frequency model

## Run :- 
To regenerate the results of our paper, you just have to run the files: Our_RR_model.ipynb and our_TF_model.ipynb 

## 📄 Citation  
If you use this repository in your research, please cite our paper:

```bibtex
@article{kumar2024quis,
  title={'Quis custodiet ipsos custodes?'Who will watch the watchmen? On Detecting AI-generated peer-reviews},
  author={Kumar, Sandeep and Sahu, Mohit and Gacche, Vardhan and Ghosal, Tirthankar and Ekbal, Asif},
  journal={arXiv preprint arXiv:2410.09770},
  year={2024}
}

📧 Contact
For questions or feedback, please contact sandeep.kumar82945@gmail.com.
