"""
A server that provides OpenAI-compatible RESTful APIs. It supports:
- Completions.
"""

import argparse
import asyncio
import json
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from vllm.utils import random_uuid
from openai import OpenAI
import httpx


from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length, is_partial_stop

# TODO: add logger 
app = FastAPI()

class OpenAIWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        proxy_url: str,
        api_key: str,
        worker_id: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            proxy_url,
            api_key,
            worker_id,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: Openai api proxy worker..."
        )

        # client = OpenAI(
        #     api_key=self.api_key,
        #     base_url=self.proxy_url,
        # )

        #TODO: Do we need Tokenizer?
        #TODO: How to handle context length?

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1


        prompt = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        #TODO: Should we handle eos_token_id?
        # if self.tokenizer.eos_token_id is not None:
        #     stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)

        request = params.get("request", None)

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        # for tid in stop_token_ids:
        #     if tid is not None:
        #         s = self.tokenizer.decode(tid)
        #         if s != "":
        #             stop.add(s)

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        gen_params = {
            "model": self.model_names[0],
            "prompt": prompt,
            "temperature": temperature,
            # "logprobs": logprobs, TODO: Should we handle logprobs?
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_new_tokens": max_new_tokens,
            "echo": echo,
            "stop_token_ids": stop_token_ids,
            "stop": list(stop),
            "stream": True,
    }
        if best_of:
            gen_params.update({"best_of": best_of})
        if use_beam_search:
            gen_params.update({"use_beam_search": use_beam_search})
        
        logger.debug(f"==== request ====\n{gen_params}")
        # return gen_params #TODO: should handle gen_params as a function


        # If you have auth, add it here (e.g., self.api_key)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if hasattr(self, "api_key") and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Forward the request to the proxy target (OpenAI-compatible backend)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                self.proxy_url, #TODO: change base_url name to proxy_url
                headers=headers,
                json=gen_params,
            ) as resp:
                if resp.status_code != 200:
                    # Optionally, yield an error message in OpenAI format
                    yield json.dumps({
                        "error": {
                            "message": f"Proxy request failed: {resp.status_code} {resp.reason_phrase}",
                            "code": resp.status_code
                        }
                    }).encode()
                    return

                # Stream and yield each event as received from the backend
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        # Optionally filter/modify here if needed
                        yield (data + "\0").encode()

        #TODO: make sure the yield here is synced with the following yield and then remove it.

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

# TODO: can I remove this completely?
def create_background_tasks(request_id):
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
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
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
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}

# TODO: do we need this also?
# @app.post("/v1/completions", dependencies=[Depends(check_api_key)])
# def create_completion(request: Request):
#     pass

# TODO: do we need this also?
# @app.post("/api/v1/chat/completions")
# def create_chat_completion(request: Request):
#     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--proxy-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
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

    args = parser.parse_args()
    
    worker = OpenAIWorker(
        args.controller_address,
        args.worker_address,
        args.api_base,
        args.api_key,
        worker_id,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")