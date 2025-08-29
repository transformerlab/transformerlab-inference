"""
A server that provides OpenAI-compatible RESTful APIs. It supports:
- Completions.
- Chat Completions.
"""

import argparse
import asyncio
import os
import base64
import shutil
from uuid import uuid4
from pathlib import Path
import json
from typing import List
import tiktoken

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastchat.constants import TEMP_IMAGE_DIR
from fastchat.utils import get_config, get_context_length
import uvicorn
import uuid
import httpx


from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)

# TODO: add logger 
app = FastAPI()

class OpenAIWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        proxy_url: str,
        api_key: str,
        proxy_model: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
        context_len: int,
        image_payload_encoding: str = "file_url",
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            no_register,
            conv_template,
        )

        self.proxy_url = proxy_url
        self.api_key = api_key
        self.proxy_model = proxy_model
        self.context_len = context_len
        self.temp_img_dir = TEMP_IMAGE_DIR
        self.model_path = model_path
        self.image_payload_encoding = image_payload_encoding

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: Openai api proxy worker..."
        )

        if not context_len:
            try:
                config = get_config(self.model_path, trust_remote_code=True)
                self.context_len = get_context_length(config)
            except Exception:
                self.context_len = 4096
        logger.info(f"Context length: {self.context_len}")

        if not no_register:
            self.init_heart_beat()

# Required param: "type" must be either "completion" or "chat-completion"
    async def generate_stream(self, params):
        self.call_ct += 1
        
        type_ = params.get("type", "completion")
        stop_str = params.get("stop", None)
        best_of = params.get("best_of", None)
        
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)
        
        images = params.get("images", [])
        image_paths = []
        if images and self.image_payload_encoding == "file_url":
            if not self.temp_img_dir:
                raise ValueError(f"Temporary image directory (`temp_img_dir`) is not set. Please provide a valid path.")
            if os.path.exists(self.temp_img_dir):
                shutil.rmtree(self.temp_img_dir)
            os.makedirs(self.temp_img_dir, exist_ok=True)

            # Decode base64 images and save them to temporary directory
            for i, b64_img in enumerate(images):
                header, encoded = b64_img.split(",", 1)
                ext = header.split("/")[1].split(";")[0]
                img_data = base64.b64decode(encoded)
                img_path = os.path.join(self.temp_img_dir, f"{uuid4()}-image_{i}.{ext}")
                with open(img_path, "wb") as f:
                    f.write(img_data)
                image_paths.append(img_path)


        
        #TODO: Should we handle logprobs?
        gen_params = {
            "model": self.proxy_model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": float(params.get("presence_penalty", 0.0)),
            "frequency_penalty": float(params.get("frequency_penalty", 0.0)),
            "max_tokens": params.get("max_new_tokens", 256),
            "stop": list(stop),
            "stream": True,
        }
        
        # Pass tools through to backend if provided
        if "tools" in params and params["tools"]:
            # Tools are already in OpenAI format by default, so pass through directly
            gen_params["tools"] = params["tools"]
        

        if type_ == "chat_completion":
            proxy_url = self.proxy_url + "/chat/completions"

            messages_to_process = params["messages"]

            if image_paths:
                for i, message in enumerate(messages_to_process):
                    if message["role"] == "user" and isinstance(message["content"], list):
                        new_content = []
                        for part in message["content"]:
                            if part.get("type") == "image_url":
                                if image_paths and self.image_payload_encoding == "file_url":
                                    image_path = image_paths.pop(0)
                                    new_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"file://{image_path}"
                                        }
                                    })
                                elif self.image_payload_encoding == "base64":
                                    image_path = images.pop(0)
                                    new_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_path
                                        }
                                    })


                            else:
                                new_content.append(part)
                        messages_to_process[i] = {
                            **message,
                            "content": new_content
                        }
            gen_params.update({
                "messages": messages_to_process
            })
        
        elif type_ == "completion":
            proxy_url = self.proxy_url + "/completions"
            gen_params.update({
                "prompt": params["prompt"],
                "echo": params.get("echo", True),
            })
            if best_of:
                gen_params.update({"best_of": best_of})

        else:
            raise ValueError(f"Unsupported type: {params['type']}")
        
        logger.info(f"==== request ====\n{gen_params}")

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        text_outputs = ""
        finish_reasons = []

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                proxy_url,
                headers=headers,
                json=gen_params,
            ) as resp:
                if resp.status_code != 200:
                    # Optionally, yield an error message in OpenAI format
                    yield json.dumps({
                        "error_code": 1,
                        "text": f"Proxy request failed: {resp.status_code} {resp.reason_phrase}"
                    }).encode()
                    return

                # Stream and yield each event as received from the backend
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if type_ == "chat_completion":
                                text = chunk["choices"][0]["delta"]["content"]
                            else:
                                text = chunk["choices"][0]["text"]
                            finish_reason = chunk.get("choices", [{}])[0].get("finish_reason", None)

                            text_outputs += text
                            if finish_reason:
                                finish_reasons.append(finish_reason)
                            
                            ret = {
                            "text": text_outputs,
                            "error_code": 0,
                            "finish_reason": finish_reasons[0] if finish_reasons else None,
                        }
                            
                            yield (json.dumps(ret) + "\0").encode()

                        except Exception:
                            print("⚠️ Failed to decode chunk:", data)
                            continue

                #yield (json.dumps({**ret, **{"finish_reason": None}}) + "\0").encode()

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()

def create_background_tasks(request_id):
    #TODO: implement this
    async def abort_request() -> None:
        print("trying to abort but not implemented")

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = str(uuid.uuid4())
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = str(uuid.uuid4())
    params["request_id"] = request_id
    params["request"] = request
    output = await worker.generate(params)
    release_worker_semaphore()
    # await engine.abort(request_id)
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    #TODO: needed to implemet this for "messages" which is for chat completion
    prompt = params["prompt"]
    #TODO: encoding = tiktoken.model.encoding_for_model(model_name)
    encoding = tiktoken.get_encoding("cl100k_base")
    input_ids = encoding.encode(prompt)
    input_echo_len = len(input_ids)
    ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
    return ret


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--proxy-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="llama3.2")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument(
        "--temp-img-dir", type=str, default=None
    )
    parser.add_argument(
        "--image-payload-encoding", type=str, choices=["file_url", "base64"], default="file_url"
    )

    args = parser.parse_args()
    
    worker = OpenAIWorker(
        args.controller_address,
        args.worker_address,
        args.proxy_url,
        args.api_key,
        args.model,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        args.context_len,
        args.image_payload_encoding,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")