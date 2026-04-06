from __init__ import logger, get_args
from config import CFG

from openai import OpenAI

import time
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

from torchinfo import summary

from vllm import LLM, SamplingParams

def hello_from_hf(messages):
        # Load model
    logger.info(f"{CFG.DEVICE_MAP=}")
    
    logger.info("Building processor...")
    processor = AutoProcessor.from_pretrained(CFG.MODEL_ID)
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CFG.MODEL_ID,
        dtype="auto",
        device_map=CFG.DEVICE_MAP
    )

    summary(model)

    logger.info("Preparing inputs...")
    
    logger.info("Applying chat template...")
    # Process input
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    logger.info("Generating outputs...")
    start = time.time()
    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=CFG.MAX_NEW_TOKENS)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    # Parse output
    logger.info("Parsing outputs...")
    out = processor.parse_response(response)
    
    duration = time.time() - start

    print(out["content"])
    print(f"{duration=} s")


def hello_from_vllm(messages, offline:bool=True):
    if offline:
        logger.info("Building tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_ID, trust_remote_code=True)

        logger.info("Building LLM...")
        llm = LLM(
            model=CFG.MODEL_ID,
            # tensor_parallel_size=2,
            max_model_len=CFG.MODEL_LEN,
            # gpu_memory_utilization=0.90,
            trust_remote_code=True
        )

        logger.info("Preparing inputs...")
        prompt = tokenizer.apply_chat_template(
            messages=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        logger.info("generating outputs...")
        outputs = llm.generate(
            prompt, 
            SamplingParams(temperature=0.15, max_tokens=CFG.MAX_NEW_TOKENS)
        )

        print(outputs[0].outputs[0].text)

    else:
        logger.info("Creating client...")
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY"
        )

        logger.info("Prompting LLM...")
        response = client.chat.completions.create(
            model=CFG.MODEL_ID,
            messages=messages,
            max_tokens=CFG.MAX_NEW_TOKENS,
            temperature=0.15
        )

        print(response.choices[0].message.content)

if __name__=="__main__":
    args = get_args()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short joke about saving RAM and running LLMs on CPU...but in french"},
    ]
    if args.vllm:
        hello_from_vllm(messages=messages, offline=False)
    else:
        hello_from_hf(messages=messages)

