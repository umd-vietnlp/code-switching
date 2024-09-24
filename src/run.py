from src.llm_engine import LLMEngine
import evaluate
import os
import asyncio
import json
from tqdm.auto import tqdm


PROMPT = """Translate the following {source} sentences to pure {target} line by line. Do not output any additional text other than the translations:
{sentence}"""

flores_mapping = {
    'flores/de-tr': {
        'source': ('German-Turkish', 'de_tr'),
        'target': ('English', 'en')
    },
    'flores/en-zh': {
        'source': ('English-Chinese', 'en_zh'),
        'target': ('Chinese', 'zh')
    },
    'flores/fr-it': {
        'source': ('French-Italian', 'fr_it'),
        'target': ('Japanese', 'jp')
    },
    'flores/ta-en': {
        'source': ('Tamil-English', 'ta_en'),
        'target': ('Czech', 'cz')
    },
    'hindi_english': {
        'source': ('Hindi-English', 'hi_en'),
        'target': ('English', 'en')
    }
}
sacrebleu = evaluate.load("sacrebleu")
MAX_BULK = 10


async def translate(engine: LLMEngine, model_name: str, num_test: int = -1):
    score_dict = {}
    for folder in flores_mapping:
        print('Translating', folder)
        source_str, source_file = flores_mapping[folder]['source']
        target_str, target_file = flores_mapping[folder]['target']
    
        source_sentences = []
        target_sentences = []
        with open(f'data/{folder}/{source_file}.txt', 'r') as in_file:
            for line in in_file:
                source_sentences.append(line.strip())
        with open(f'data/{folder}/{target_file}.txt', 'r') as in_file:
            for line in in_file:
                target_sentences.append(line.strip())
        assert len(target_sentences) == len(source_sentences)

        if num_test > 0:
            target_sentences = target_sentences[: num_test]
            source_sentences = source_sentences[: num_test]
    
        tasks = []
        translated_sentences = []
        for i, sent in tqdm(enumerate(source_sentences), total=len(source_sentences)):
            prompt = PROMPT.format(source=source_str, target=target_str, sentence=sent)
            task = asyncio.create_task(
                engine.agenerate([{'role': 'user', 'content': prompt}], model=model_name)
            )
            tasks.append(task)
            if i % MAX_BULK == 0 or i == len(source_sentences)-1:
                for task in tasks:
                    result = await task
                    translated_sentences.append(result)
                tasks = []

        score = sacrebleu.compute(
            predictions=[pred.lower() for pred in translated_sentences],
            references=[target.lower() for target in target_sentences]
        )
        score_dict[folder] = score
        print(f'{folder} - BLEU socre:', score['score'])
    return score_dict


async def run():
    model_names = [
        ('openai', 'gpt-4o'),
        ('openai', 'gpt-3.5-turbo'),
        # ('fireworks', 'accounts/fireworks/models/llama-v3-70b-instruct-hf')
    ]
    api_keys = {
        'openai': 'sk-proj-xxx'
    }
    tasks = []
    for provider, model in model_names:
        engine = LLMEngine(provider=provider,
                           api_key=api_keys[provider])
        tasks.append((model, asyncio.create_task(translate(engine, model))))

    os.makedirs('results/', exist_ok=True)
    print(f'========== EVALUATION ===========')
    for model, task in tasks:
        print('Model:', model)
        result = await task
        print(result)
        print('=====================')
        with open(f'results/{model}.json', 'w') as out_f:
            json.dump(result, out_f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    asyncio.run(run())
