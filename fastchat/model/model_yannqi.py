from threading import Thread
import gc
import torch
import requests
from PIL import Image
from transformers import TextIteratorStreamer, AutoProcessor

def generate_stream_yannqi(
    model,
    tokenizer,
    params,
    device,
    context_len,
    stream_interval=2,
    judge_sent_end=False
):
    """Custom generate stream function for YannQi R-4B models"""
    # Get parameters from the request
    prompt = params.get("prompt", "")
    messages = params.get("messages", None)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    min_p = float(params.get("min_p", 0.0))
    max_new_tokens = int(params.get("max_new_tokens", 16384))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    thinking_mode = params.get("thinking_mode", "auto")
    images = params.get("images", None)  # List of image URLs or PIL Images


    # Get model name from model object if available
    model_name = getattr(model, 'name_or_path', None) or params.get("model", None)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


    if processor.tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(processor.tokenizer.eos_token_id)

    # Process images if provided
    processed_images = None
    if images:
        processed_images = []
        for img in images:
            if isinstance(img, str):  # URL
                try:
                    processed_images.append(Image.open(requests.get(img, stream=True).raw))
                except Exception as e:
                    print(f"Error loading image from URL {img}: {e}")
                    continue
            elif isinstance(img, Image.Image):  # PIL Image
                processed_images.append(img)
        
        # Use first image if multiple provided (model might support only one)
        processed_images = processed_images[0] if processed_images else None

    # Format input based on whether we have messages or a plain prompt
    if messages:
        # Add the type text field to all message contents if missing
        for message in messages:
            if isinstance(message.get("content"), str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            elif isinstance(message.get("content"), list):
                for content in message["content"]:
                    if "type" not in content and "text" in content:
                        content["type"] = "text"
        # Apply chat template with thinking mode
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode=thinking_mode
        )
    else:
        # Create message format from plain prompt
        if processed_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_images},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode="auto"
        )


    # Process inputs
    if processed_images:
        inputs = processor(
            images=processed_images,
            text=text,
            return_tensors="pt"
        ).to(device)
    else:
        print("PROCESSING INPUTS")
        inputs = processor(
            text=text,
            return_tensors="pt"
        ).to(device)

    input_ids = inputs["input_ids"]
    input_echo_len = input_ids.shape[1]

    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else 1.0,
    }

    if top_p < 1.0:
        generate_kwargs["top_p"] = top_p
    if top_k > 0:
        generate_kwargs["top_k"] = top_k
    if repetition_penalty > 1.0:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    if min_p > 0.0:
        generate_kwargs["min_p"] = min_p

    # Add all other inputs to generation kwargs
    for key, value in inputs.items():
        if key != "input_ids":
            generate_kwargs[key] = value

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=not echo, skip_special_tokens=True)
    generate_kwargs["streamer"] = streamer

    # Start generation in a separate thread
    thread = Thread(target=lambda: model.generate(input_ids=input_ids, **generate_kwargs))
    thread.start()

    # Track generation progress
    generated_tokens = 0
    output_text = ""

    # Stream tokens
    for new_text in streamer:
        output_text += new_text
        generated_tokens += 1

        # Check for stop strings
        should_stop = False
        if stop_str:
            if isinstance(stop_str, str):
                if stop_str in output_text:
                    output_text = output_text[: output_text.find(stop_str)]
                    should_stop = True
            elif isinstance(stop_str, list):
                for stop in stop_str:
                    if stop in output_text:
                        output_text = output_text[: output_text.find(stop)]
                        should_stop = True
                        break

        # Stream at intervals or when stopping
        if generated_tokens % stream_interval == 0 or should_stop:
            yield {
                "text": output_text,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": generated_tokens,
                    "total_tokens": input_echo_len + generated_tokens,
                },
                "finish_reason": "stop" if should_stop else None,
            }

        if should_stop:
            break

    # Wait for thread to complete
    if thread.is_alive():
        thread.join(timeout=3600)

    # Final output with finish reason
    yield {
        "text": output_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": generated_tokens,
            "total_tokens": input_echo_len + generated_tokens,
        },
        "finish_reason": "length",
    }

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()
