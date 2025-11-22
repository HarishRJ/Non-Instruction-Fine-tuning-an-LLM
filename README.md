# Overview: Fine-Tuning TinyLlama for Pharmaceutical Domain Adaptation

This project demonstrates a complete pipeline for adapting **TinyLlama** (a compact 1.1B-parameter LLM) to the **pharmaceutical domain** using causal language modeling. We focus on non-instructional fine-tuning to enhance the model's ability to generate coherent, domain-specific text—such as drug interactions, clinical trial summaries, and medical abstracts—without requiring labeled Q&A pairs.

## Key Achievements

### 1. **Data Preparation & Tokenization**
   - Loaded domain-specific datasets (e.g., custom `pharma_non_instruction.jsonl` with sentences like "Metformin improves insulin sensitivity in the liver.").
   - Applied causal LM tokenization: Texts are tokenized to fixed-length sequences (512 tokens) with padding/truncation. Labels are auto-shifted copies of input IDs, enabling next-token prediction training.
   - Supported datasets: PubMed abstracts, FineWeb (filtered web text), scientific papers from ArXiv, and OpenWebText for broader medical corpus.

### 2. **Model Training Strategies**
   - **Full Fine-Tuning**: Trained the entire model over 2 epochs on tokenized data, using mixed-precision (FP16) for efficiency. This updates all parameters for deep domain infusion.
   - **Partial Fine-Tuning (Layer Freezing)**: Froze the base layers and only tuned the last 4 transformer blocks + LM head, reducing trainable parameters to ~10% while preserving general knowledge.
   - **LoRA (Low-Rank Adaptation)**: Applied efficient adapters (r=8, targeting Q/V projections) with 8-bit quantization, training just <1% of parameters over 5 epochs. This enables low-resource deployment on consumer hardware.

### 3. **Evaluation & Inference**
   - Post-training, generated completions for pharma prompts (e.g., "Clinical trials demonstrated that combining Atorvastatin with Ezetimibe" → "...significantly reduces LDL cholesterol levels...").
   - Checkpoints saved for merging adapters or resuming; inference uses greedy/sampling with temperature for natural outputs.

## Impact
By the end, we have a specialized LLM that outperforms the base TinyLlama on pharma text generation, measured informally via perplexity drops and relevance in completions. This serves as a blueprint for domain adaptation in healthcare AI, balancing performance with efficiency. Future work could integrate evaluation metrics (e.g., BLEU for medical summaries) or deploy via Hugging Face Spaces.
