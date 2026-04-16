# char_causal_language_modeling

Character-level causal language model in Python (PyTorch) for next-character prediction and text generation.

## Repository

GitHub: https://github.com/salyamq/char_causal_language_modeling

## Project Structure

```text
char_causal_language_modeling/
├── data_preparation.py   # preparing text data for training
├── tokenizer.py          # char-level tokenizer
├── embeddings.py         # embedding logic
├── model.py              # causal language model architecture
├── learning.py           # training loop / optimization
├── test.py               # inference / testing script
├── requirements.txt      # dependencies
├── last_model.pt         # saved trained weights
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/salyamq/char_causal_language_modeling.git
cd char_causal_language_modeling

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Usage

### 1) Prepare data

```bash
python data_preparation.py
```

### 2) Train model

```bash
python learning.py
```

### 3) Run test / generation

```bash
python test.py
```

## Example 'test.py' output (may vary due to sampling):


```text
Loaded checkpoint: last_model.pt
Prompt: "The "
Max new tokens: 200

Generated text:
The gage to be the very content.--Is never
Of it booking and let frame what I would I seize your house,
Sirrah is much until. But, be removed knowledge:
The queen I send to heaven and his son,
Or, the war
```

## Notes

- Model checkpoint example: `last_model.pt`
- Main model implementation is in `model.py`
- Tokenization is character-level (`tokenizer.py`)
- Code comments in Russian
- Parameters: ~1.2M
- Type: Character-level causal Transformer
- last_model.pt was trained for 1 epoch
- Dataset: tiny_shakespeare


## License
MIT
