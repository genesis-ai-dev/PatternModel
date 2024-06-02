# Language-Agnostic Translation

## Overview

This project implements a language-agnostic translation method that focuses on pattern matching rather than direct translation. The key components of this system are:

1. **GlobalRelativeTokenizer**: A `tokenizer` that assigns a unique numerical value to each unique word in the corpus, creating a language-agnostic representation of the text.
2. **GPT4MetaTranslator and ClaudeMetaTranslator**: LLMs that use the GlobalRelativeTokenizer target sequences into numerical tokens, allowing them to find patterns in the data and generate target sequences for new source inputs.
3. **SilverPath**: A module that analyzes the existing data and provides a metric for how confident the model should be in its translations, as well as a means of determining which materials might be highest priority for further curation.
4. **Translator**: A module that combines the components to perform automated translation, leveraging the linguistic anomaly detection to provide quality scores for the generated translations.

## Why this approach forces pattern matching

The key innovation of this approach is the use of the GlobalRelativeTokenizer, which converts the target texts into a language-agnostic numerical representation. This forces the translation models to focus on finding patterns in the data, rather than relying on direct word-to-word translations. It also allows for an easy way to spot hallucinations, only contextual numerical values should appear in the output. Supprisingly, when the model doesn't have good examples, it will hallucinate a new number, these can be replaced with an [unclear] marker. 

By representing the text as a sequence of numerical tokens, the models are forced to learn the underlying linguistic structures and mappings between the source and target sequences, rather than simply memorizing specific word-level translations, or relying on any outside biases. This allows the models to generalize better to new inputs, as they can apply the learned patterns to generate appropriate target sequences. Hypothetically this means that it is equally good at translating to almost any language, which also makes it easier to test.

## SilverPath vs GoldenPath
The `SilverPath` module further reinforces this pattern-matching approach by providing a metric for the model's confidence in its translations. Based on existing data, it is able to determine how easy it will be for a model to translate a given text. This helps the user understand which prompts are likely to be successfully translated based on the existing data, and where further curation may be needed to improve the model's performance. `GoldenPath` techniques are generally the opposite, they take a collection of prompts and tell people what samples they should get. `SilverPath` can be useful if there are existing translation samples, and you want to make the most of what you have so far -- even if they don't align with a specific `GoldenPath` for all desired prompts.

## How to use.

To use this method, you'll need to follow these steps:

1. Prepare your data: Ensure that you have three text files in the specified data directory: `source.txt`, `target.txt`, and `prompts.txt`. These files should contain the source sequences, target sequences, and prompts to be translated, respectively. They should be aligned by line.
2. Set up the API keys: Provide the necessary API keys for the translation services (OpenAI or Anthropic) in the `ClaudeMetaTranslator` and `GPT4MetaTranslator` classes.
3. (optional) Run the `SilverPath` module: This will analyze the existing data and generate a `ranked_prompts.jsonl` file, which provides a ranking of the prompts based on how well they can be translated using the current data.
4. Run the `Translator` module: This will perform the automated translation, using the `ClaudeMetaTranslator` and the linguistic anomaly detection to provide quality scores for the generated translations. The output will be written to an `output.txt` file in the data directory.

By following this process, you can leverage the language-agnostic pattern matching capabilities of the translation models to perform high-quality, automated translation, while also gaining insights into the strengths and weaknesses of your dataset.

## Linguistic Anomaly Detection (LAD)

The `lad.py` module introduces the `LinguisticAnomalyDetector` class, which is responsible for performing Linguistic Anomaly Detection on the translated texts. The key aspects of this class are:

1. **Initialization**: The class takes two lists as input, `source_reference` and `target_reference`, which represent the reference texts for the source and target languages, respectively.
2. **Ranking Similarity**: The `_rank_similar` method ranks the given references by their similarity to a query text using the `SequenceMatcher` from the `difflib` module.
3. **Anomaly Detection**: The `detect` method compares the ranking of the target draft against the target references and the ranking of the source baseline against the source references, calculating a similarity score between the two rankings.
4. **Score Tracking**: The `detect` method keeps track of the individual similarity scores in the `self.scores` list, which can be used to calculate an average score across multiple detections.

The `LinguisticAnomalyDetector` class is integrated into the `Translator` module to provide quality scores for the generated translations, helping the user identify areas where the model's performance may be lacking, this should be even more useful when combined with the scores generated by `SilverPath`?