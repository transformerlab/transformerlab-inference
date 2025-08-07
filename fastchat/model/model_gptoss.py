import gc
import torch

try:
    from openai_harmony import (
        load_harmony_encoding,
        StreamableParser,
        HarmonyEncodingName,
        Role
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    print("Warning: openai_harmony package not available. GPT-OSS streaming may not work correctly.")

def generate_stream_gptoss(
    model,
    tokenizer,
    params,
    device,
    context_len,
    stream_interval=2,
    judge_sent_end=False
):
    """Custom generate stream function for GPT-OSS models using openai_harmony decoder"""
    # Get parameters from the request
    prompt = params.get("prompt", "")
    messages = params.get("messages", None)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    min_p = float(params.get("min_p", 0.0))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    model_name = params.get("model", None)

    print(f"Generating with GPT-OSS model: {model_name}")

    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    # GPT-OSS models are instruction-tuned, so we'll use chat template by default
    is_base_model = "base" in model_name.lower() if model_name else False

    if not is_base_model:
        # Format input based on whether we have messages or a plain prompt
        if messages:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
        else:
            # Convert plain prompt to message format for chat template
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
    else:
        # For base models, use plain tokenization
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    input_echo_len = input_ids.shape[1]
    
    # Create attention mask if not present (needed when pad_token_id == eos_token_id)
    if "attention_mask" not in inputs:
        attention_mask = torch.ones_like(input_ids)
        inputs["attention_mask"] = attention_mask

    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else 1.0,
        "pad_token_id": tokenizer.eos_token_id,  # GPT-OSS may need explicit pad token
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    if top_p < 1.0:
        generate_kwargs["top_p"] = top_p
    if top_k > 0:
        generate_kwargs["top_k"] = top_k
    if repetition_penalty > 1.0:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    if min_p > 0.0:
        generate_kwargs["min_p"] = min_p

    # Add stop token ids
    if stop_token_ids:
        generate_kwargs["eos_token_id"] = stop_token_ids

    if not HARMONY_AVAILABLE:
        # Fallback to regular streaming if harmony is not available
        from transformers import TextIteratorStreamer
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs["streamer"] = streamer
        
        # Start generation in a separate thread
        from threading import Thread
        thread = Thread(target=lambda: model.generate(**inputs, **generate_kwargs))
        thread.start()
        
        generated_tokens = 0
        output_text = ""
        
        for new_text in streamer:
            output_text += new_text
            generated_tokens += 1
            
            # Check for stop strings
            should_stop = False
            if stop_str:
                if isinstance(stop_str, str):
                    if stop_str in output_text:
                        output_text = output_text[:output_text.find(stop_str)]
                        should_stop = True
                elif isinstance(stop_str, list):
                    for stop in stop_str:
                        if stop in output_text:
                            output_text = output_text[:output_text.find(stop)]
                            should_stop = True
                            break
            
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
        
        if thread.is_alive():
            thread.join(timeout=3600)
        
        yield {
            "text": output_text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": generated_tokens,
                "total_tokens": input_echo_len + generated_tokens,
            },
            "finish_reason": "length",
        }
        return

    # Use openai_harmony with streaming
    try:
        # Create a custom streamer that captures tokens for harmony processing
        class HarmonyTokenStreamer:
            def __init__(self, tokenizer, harmony_encoding):
                self.tokenizer = tokenizer
                self.stream_parser = StreamableParser(harmony_encoding, role=Role.ASSISTANT)
                self.tokens = []
                self.finished = False
                
            def put(self, value):
                if value is None:
                    self.finished = True
                    return
                
                # Handle different types of input
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        # Single token tensor
                        token = value.item()
                        self.tokens.append(token)
                    elif value.numel() > 1:
                        # Multiple tokens (like input_ids) - flatten and add all
                        tokens = value.flatten().tolist()
                        self.tokens.extend(tokens)
                else:
                    # Already a scalar or list
                    if isinstance(value, (list, tuple)):
                        self.tokens.extend(value)
                    else:
                        self.tokens.append(value)
                
            def end(self):
                self.finished = True
        
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        harmony_streamer = HarmonyTokenStreamer(tokenizer, encoding)
        generate_kwargs["streamer"] = harmony_streamer
        
        # Start generation in a separate thread
        from threading import Thread
        thread = Thread(target=lambda: model.generate(**inputs, **generate_kwargs))
        thread.start()
        
        # Track state for streaming
        current_output = ""
        analysis_content = ""
        final_content = ""
        in_analysis = False
        in_final = False
        analysis_started = False
        final_started = False
        generated_tokens_count = 0
        processed_tokens = 0
        should_stop = False
        
        # Stream processing
        while not harmony_streamer.finished or processed_tokens < len(harmony_streamer.tokens):
            # Process any new tokens
            while processed_tokens < len(harmony_streamer.tokens):
                token = harmony_streamer.tokens[processed_tokens]
                harmony_streamer.stream_parser.process(token)
                processed_tokens += 1
                generated_tokens_count += 1
                
                # Check current state
                current_role = harmony_streamer.stream_parser.current_role
                current_channel = harmony_streamer.stream_parser.current_channel
                last_content_delta = harmony_streamer.stream_parser.last_content_delta
                
                # Handle channel transitions
                if current_role == Role.ASSISTANT and current_channel == "analysis":
                    if not analysis_started:
                        analysis_started = True
                        in_analysis = True
                        current_output += "<think>"
                    
                    if last_content_delta and in_analysis:
                        analysis_content += last_content_delta
                        current_output += last_content_delta
                        
                elif current_role == Role.ASSISTANT and current_channel == "final":
                    if in_analysis and not final_started:
                        # Transition from analysis to final
                        current_output += "</think>"
                        in_analysis = False
                        final_started = True
                        in_final = True
                    
                    if last_content_delta and in_final:
                        final_content += last_content_delta
                        current_output += last_content_delta
                
                # Check for stop strings in the current output
                should_stop = False
                if stop_str:
                    if isinstance(stop_str, str):
                        if stop_str in current_output:
                            current_output = current_output[:current_output.find(stop_str)]
                            should_stop = True
                    elif isinstance(stop_str, list):
                        for stop in stop_str:
                            if stop in current_output:
                                current_output = current_output[:current_output.find(stop)]
                                should_stop = True
                                break
                
                # Stream at intervals or when stopping
                if generated_tokens_count % stream_interval == 0 or should_stop:
                    yield {
                        "text": current_output,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": generated_tokens_count,
                            "total_tokens": input_echo_len + generated_tokens_count,
                        },
                        "finish_reason": "stop" if should_stop else None,
                    }
                
                if should_stop:
                    break
            
            if should_stop:
                break
                
            # Small sleep to avoid busy waiting
            import time
            time.sleep(0.001)
        
        # Wait for generation to complete
        if thread.is_alive():
            thread.join(timeout=3600)
        
        # Final output - ensure we close any open thinking tags
        if in_analysis and not final_started:
            current_output += "</think>"
        
        yield {
            "text": current_output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": generated_tokens_count,
                "total_tokens": input_echo_len + generated_tokens_count,
            },
            "finish_reason": "length",
        }
        
    except Exception as e:
        print(f"Error using openai_harmony decoder: {e}")
        # Fallback to regular streaming if harmony fails
        from transformers import TextIteratorStreamer
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs["streamer"] = streamer
        
        # Remove harmony-specific kwargs that might cause issues
        if "return_dict_in_generate" in generate_kwargs:
            del generate_kwargs["return_dict_in_generate"]
        if "output_scores" in generate_kwargs:
            del generate_kwargs["output_scores"]
        
        # Start generation in a separate thread
        from threading import Thread
        thread = Thread(target=lambda: model.generate(**inputs, **generate_kwargs))
        thread.start()
        
        generated_tokens = 0
        output_text = ""
        
        for new_text in streamer:
            output_text += new_text
            generated_tokens += 1
            
            # Check for stop strings
            should_stop = False
            if stop_str:
                if isinstance(stop_str, str):
                    if stop_str in output_text:
                        output_text = output_text[:output_text.find(stop_str)]
                        should_stop = True
                elif isinstance(stop_str, list):
                    for stop in stop_str:
                        if stop in output_text:
                            output_text = output_text[:output_text.find(stop)]
                            should_stop = True
                            break
            
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
        
        if thread.is_alive():
            thread.join(timeout=3600)
        
        yield {
            "text": output_text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": generated_tokens,
                "total_tokens": input_echo_len + generated_tokens,
            },
            "finish_reason": "length",
        }

    # Clean up - especially important for GPT-OSS with tensor parallelism
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()
