import os
import argparse
import base64
from tqdm import tqdm
import asyncio
from io import BytesIO
from PIL import Image, ImageFile
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from typing import Dict, List
import io
import random
from sglang import Engine
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoConfig, AutoTokenizer
import torch


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

TILE_SIZE = (4096, 4096)
TILE_XY = (0, 0)
MAX_PIXELS = 4096 * 4096

LABEL_MAP = {
    0: "非鼻咽癌",
    1: "鼻咽癌"
}

def get_parser():
    parser = argparse.ArgumentParser(description="API Test for PrePATH")
    parser.add_argument("--model_api_key", default="", help="OpenAI API key")
    parser.add_argument("--model_name", default="gemini-3-pro-preview", help="OpenAI Model Name")
    parser.add_argument('--model_base_url', type=str, default=None)
    parser.add_argument("--include_thoughts", default="True", type=str, help="Whether to include model thoughts in the response")
    parser.add_argument('--thinking_budget', type=str, default="low")
    parser.add_argument('--model_max_concurrency', type=int, default=1)
    parser.add_argument("--data_path", default="data", help="Directory containing test images")
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--use_async', type=str, default="True")
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--output_path', type=str, default="./results.jsonl")
    parser.add_argument('--dnn_results', type=str, default="")
    parser.add_argument('--use_fewshot', type=str, default="true")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--default_system_prompt', type=str, default="")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.1)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--cuda_visible_devices', type=str, default="0")
    parser.add_argument('--tensor_parallel_size', type=str, default="8")
    parser.add_argument('--data_parallel_size', type=str, default="1")
    parser.add_argument('--split', type=str, default="false")
    parser.add_argument('--model_type', type=str, default="qwen3")
    parser.add_argument('--enable_thinking', type=str, default="true")
    parser.add_argument('--max_concurrency', type=int, default=1)

    return parser


def split_image_to_tiles(image_path, tile_size=(4096, 4096)):
    """
    将一张大图切成多个 tile
    Args:
        image_path: png/jpg 路径
        tile_size: (width, height) 每个 tile 的尺寸

    Returns:
        List of PIL.Image tiles
    """
    tiles = []
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        W, H = img.size
        tile_w, tile_h = tile_size

        for y in range(0, H, tile_h):
            for x in range(0, W, tile_w):
                right = min(x + tile_w, W)
                lower = min(y + tile_h, H)
                tile = img.crop((x, y, right, lower))
                tiles.append(tile)
    return tiles


def get_mime_type(file_path: str) -> str:
    """
    根据文件路径或文件名返回对应的 MIME Type。
    如果扩展名不在映射表中，返回 None。
    """
    ext_to_mime = {
        ".bmp": "image/bmp",
        ".jpe": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
        ".heic": "image/heic",
    }
    ext = os.path.splitext(file_path)[1].lower()  # 获取后缀并转小写
    return ext_to_mime.get(ext, None)


def is_blank_tile(
    pil_img,
    tissue_ratio_threshold=0.05,
    sat_threshold=15
):
    import numpy as np
    import cv2

    img = np.array(pil_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]

    tissue_mask = saturation > sat_threshold
    tissue_ratio = tissue_mask.sum() / tissue_mask.size

    is_blank = tissue_ratio < tissue_ratio_threshold
    return is_blank, tissue_ratio


def encode_image(image, tile_size=(4096, 4096), jpeg_quality=90, split:bool=True):
    if isinstance(image, str):
        # TEMP_DIR = "./temp_blank_tiles"
        # os.makedirs(TEMP_DIR, exist_ok=True)
        try:
            with Image.open(image) as img:
                w, h = img.size

            if w * h <= MAX_PIXELS:
                with open(image, "rb") as image_file:
                    return (
                        base64.b64encode(image_file.read()).decode("utf-8"),
                        get_mime_type(image)
                    )

            if split:
                tiles = split_image_to_tiles(image, tile_size)
            else:
                tiles = [Image.open(image)]
            b64_list = []

            # image_name = os.path.splitext(os.path.basename(image))[0]

            for idx, tile in enumerate(tiles):
                # is_blank, tissue_ratio = is_blank_tile(
                #     tile, tissue_ratio_threshold=0.05
                # )

                # if is_blank:
                #     # save_name = (
                #     #     f"{image_name}_tile{idx:03d}"
                #     #     f"_tissue{tissue_ratio:.3f}.jpg"
                #     # )
                #     # save_path = os.path.join(TEMP_DIR, save_name)
                #     # tile.save(save_path, "JPEG", quality=90)
                #     continue
            
                buffer = io.BytesIO()
                tile.save(buffer, "JPEG", quality=jpeg_quality, optimize=True)
                b64_list.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

            # if len(b64_list) == 0:
            #     mid = len(tiles) // 2
            #     buffer = io.BytesIO()
            #     tiles[mid].save(buffer, "JPEG", quality=jpeg_quality, optimize=True)
            #     b64_list.append(
            #         base64.b64encode(buffer.getvalue()).decode("utf-8")
            #     )
                
            return b64_list, "image/jpeg"

        except Exception as e:
            raise RuntimeError(f"encode_image failed for {image}: {e}")
        
    elif isinstance(image, Image.Image):
        buffered = BytesIO()
        fmt = image.format or "PNG"
        image.save(buffered, format=fmt)
        return base64.b64encode(buffered.getvalue()).decode("utf-8"), fmt
    else:
        raise TypeError("Input must be a file path or a PIL.Image object")


class API_MODEL:
    def __init__(self, args):
        if args.use_async.lower() in ["true", "True"]:
            self.async_client = AsyncOpenAI(
                api_key=args.model_api_key,
                base_url=args.model_base_url,
            )
        else:
            self.client = OpenAI(
                api_key=args.model_api_key,
                base_url=args.model_base_url,
            )
            
        self.model_name = args.model_name
        self.include_thoughts = args.include_thoughts.lower() in ["true", "True"]
        self.thinking_budget = args.thinking_budget
        self.max_concurrency = args.model_max_concurrency
        self.max_completion_tokens=args.max_new_tokens
        self.num_samples = args.num_samples
        self.use_fewshot = args.use_fewshot.lower() == "true"
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.data = self.read_data()
        self.idx_list = []

        self.dnn_results = {}

        if args.dnn_results:
            self.read_dnn_results(args.dnn_results)

        self.fewshot_images = self.get_fewshot_images() if self.use_fewshot else []
        self.messages_list = self.process_messages()
        
        print("API Model:", self.model_name)
        print("Concurrency", self.max_concurrency)

        # import pdb; pdb.set_trace()
    
    def read_dnn_results(self, dnn_results):
        df = pd.read_excel(dnn_results)

        for _, row in df.iterrows():
            slide_path = row["slide_path"]
            label = int(row["label"])
            pred = int(row["pred"])
            prob = float(row["prob"])

            slide_name = os.path.basename(slide_path)
            slide_id = os.path.splitext(slide_name)[0]

            self.dnn_results[slide_id] = {
                "label": label,
                "pred": pred,
                "prob": prob if pred == 1 else 1-prob
            }
    
    def read_data(self):
        df = pd.read_csv(self.data_path)

        if self.num_samples > 0:
            df = df.iloc[:self.num_samples]

        data = []
        for idx, row in df.iterrows():
            data.append({
                "id": idx,
                "image_path": row["file_path"],
                "label": int(row["label"])
            })

        return data
    
    def get_fewshot_images(self):
        images_rootdir = "/zju_0038/medical/workspace/PrePATH/api_test/fewshot_images"
        images = ["672304.png", "683520.png", "671393.png", "691683.jpg"]
        fewshot_images_prompt = []
        for image in images:
            base64_image, mime_name = encode_image(os.path.join(images_rootdir, image))
            if not isinstance(base64_image, List):
                base64_image = [base64_image]
            for bi in base64_image:
                fewshot_images_prompt.append({"type": "image_url", "image_url": {"url": f"data:{mime_name};base64,{bi}"}})
        return fewshot_images_prompt

    def process_messages(self):
        existing = self.read_jsonl(self.output_path)
        print(f"[Resume] Loaded {len(existing)} previous results from {self.output_path}")

        new_messages = []
        for data in tqdm(self.data, desc="Split Image ..."):
            self.idx_list.append(data["id"])
            if data["id"] in existing:
                new_messages.append({})
                continue
            image_path = data["image_path"]
            slide_id = os.path.basename(os.path.dirname(image_path))
            prompt = self.get_prompt(slide_id)
            base64_image, mime_name = encode_image(image_path)
            fewshot_images = self.fewshot_images[:]
            if not isinstance(base64_image, List):
                base64_image = [base64_image]
            for bi in base64_image:
                fewshot_images.extend([{"type": "image_url", "image_url": {"url": f"data:{mime_name};base64,{bi}"}}])
            fewshot_images.extend([{"type": "text", "text": prompt}])
            new_messages.append({"messages": [{"role": "user", "content": fewshot_images}], "label": data["label"], "file_path": image_path, "tiles": len(base64_image), "prompt": prompt})

        return new_messages

    def generate_output(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages["messages"],
                extra_body = {
                    "thinking_budget": self.thinking_budget
                }
                # extra_body={
                #     "extra_body": {
                #         "google": {
                #             "thinking_config": {
                #                 "include_thoughts": self.include_thoughts,
                #                 "thinking_budget": self.thinking_budget
                #             }
                #         }
                #     }
                # }
            )
            for chunk in completion:
                if not completion.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                    content = f"{chunk.usage}"
                else:
                    content = completion.choices[0].message.content

            return f"<answer>{content}</answer>"

        except Exception as e:
            print("Error", e)
            return f"<answer>Error: {e}</answer>"

    def generate_outputs(self, output_path: str = ""):
        existing = self.read_jsonl(output_path)
        print(f"[Resume] Loaded {len(existing)} previous results from {output_path}")

        res = []
        for idx, messages in tqdm(zip(self.idx_list, self.messages_list), total=len(self.messages_list), desc="Generate outputs"):
            if idx in existing:
                res.append(existing[idx])
                continue
            result = self.generate_output(messages)
            answer = extract(result, "answer", hard=True)
            pred = extract(answer, "class", hard=False, number=True)
            result = {"id": idx, "prompt": messages["prompt"], "response": result, "label": messages["label"], "correct": int(pred) == messages["label"] if pred != "" else False, "file_path": messages["file_path"], "tiles": messages["tiles"]}
            self.append_jsonl(output_path, result)
            res.append(result)
        return res
    

    async def async_generate_output(self, messages):
        # print("messages:", messages)

        completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages["messages"],
            extra_body = {
                "thinking_budget": self.thinking_budget
            }
            # extra_body={
            #     "extra_body": {
            #         "google": {
            #             "thinking_config": {
            #                 "include_thoughts": self.include_thoughts,
            #                 "thinking_budget": self.thinking_budget
            #             }
            #         }
            #     }
            # }
        )

        content = completion.choices[0].message.content
        return f"<answer>{content}</answer>"

    async def async_generate_outputs(self, output_path: str = ""):
        existing = self.read_jsonl(output_path)
        print(f"[Resume] Loaded {len(existing)} previous results from {output_path}")

        semaphore = asyncio.Semaphore(self.max_concurrency)
        results = [None] * len(self.messages_list)

        async def _worker(idx, sample_id, payload):
            if sample_id in existing:
                results[idx] = existing[sample_id]
                return 

            async with semaphore:
                results[idx] = await self.async_generate_output(payload)
                # write to jsonl
                answer = extract(results[idx], "answer", hard=True)
                pred = extract(answer, "class", hard=False, number=True)
                results[idx] = {"id": sample_id, "prompt": payload["prompt"], "response": results[idx], "label": payload["label"], "correct": int(pred) == payload["label"] if pred != "" else False, "file_path": payload["file_path"], "tiles": payload["tiles"]}
                self.append_jsonl(output_path, results[idx])
                
        
        tasks = [asyncio.create_task(_worker(i, sample_id, m)) for i, (sample_id, m) in enumerate(zip(self.idx_list, self.messages_list))]

        # show progress as tasks complete
        for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async generate outputs"):
            await _
        
        print(len(results))
        return results
    
    @staticmethod
    def read_jsonl(path):
        if not os.path.exists(path):
            return {}
        
        data = {}
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                idx = obj["id"]
                data[idx] = obj
        return data

    @staticmethod
    def append_jsonl(path, obj):
        with open(path, "a", encoding="utf8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def few_shot_prompt():
        pass

    def get_prompt(self, slide_id):
        pred, prob = LABEL_MAP[int(self.dnn_results[slide_id]["pred"])], self.dnn_results[slide_id]["prob"]
        # prob = random.uniform(0.8, 1.0)  # random.random()
        fewshot_prompt = """
将会提供多张图像，请按顺序理解：

第 1 张图：鼻咽癌示例（用于参考，不需要判断）
第 2 张图：鼻咽癌示例（用于参考，不需要判断）
第 3 张图：非鼻咽癌示例（用于参考，不需要判断）
第 4 张图：非鼻咽癌示例（用于参考，不需要判断）
剩下的图像：可能是一个图像的多个 tiles, 需要你判断是否为鼻咽癌。
""" if self.use_fewshot else ""
        
        prompt = f"""你是一个鼻咽癌病理图像专家。你的任务是分析提供给你的病理图像，并判断该图像是否为鼻咽癌。

{fewshot_prompt}

我们已有的判断结果是 {pred}，置信度为 {round(prob, 4)*100}%。请你基于已有的判断结果，再次进行判断

请严格按照以下格式回答：
<class>类别</class>
<reasoning>理由</reasoning>

类别说明：
请严格输出 "非鼻咽癌" 或 "鼻咽癌"，不可输出其他值或文字。

理由说明：
请详细说明你是如何从图像中得出结论的，可以参考细胞形态、核形态、组织结构、染色特点等典型病理特征。  
请注意：
- 只能基于图像信息判断，不要凭空想象。  
- 绿色曲线标注仅代表有效染色区域，不代表病变区域，请勿将其作为判断依据。"""
        return prompt


class Qwen3VL:
    def __init__(self, args):
        print("tp size:", int(os.environ.get("TENSOR_PARALLEL_SIZE", 1)))
        print("dp size:", int(os.environ.get("DATA_PARALLEL_SIZE", 1)))
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.config = AutoConfig.from_pretrained(args.model_path)
        self.context_length = self.tokenizer.model_max_length
        self.llm = Engine(
            model_path=args.model_path,
            enable_multimodal=True,
            mem_fraction_static=0.6,
            tp_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", 1)),
            dp_size=int(os.environ.get("DATA_PARALLEL_SIZE", 1)),
            attention_backend="fa3",
            context_length=self.context_length,
            # enable_torch_compile=False,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            enable_deterministic_inference=True,
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path, padding_side="right")
        self.max_concurrency = args.model_max_concurrency
        self.split = args.split.lower() == "true"

        self.default_system_prompt = args.default_system_prompt

        self.sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }

        print(f"default_system_prompt: {args.default_system_prompt}, {self.sampling_params}")

        self.num_samples = args.num_samples
        self.use_fewshot = args.use_fewshot.lower() == "true"
        self.use_dnn = args.dnn_results != ""
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.data = self.read_data()
        self.idx_list = []

        self.dnn_results = {}

        if args.dnn_results:
            self.read_dnn_results(args.dnn_results)

        self.fewshot_images = self.get_fewshot_images() if self.use_fewshot else []
        self.messages_list = self.process_messages()

    def read_dnn_results(self, dnn_results):
        df = pd.read_csv(dnn_results, encoding="utf-8")

        for _, row in df.iterrows():
            slide_path = row["slide_path"]
            label = int(row["label"])
            pred = int(row["pred"])
            prob = float(row["prob"])

            slide_name = os.path.basename(slide_path)
            slide_id = os.path.splitext(slide_name)[0]

            self.dnn_results[slide_id] = {
                "label": label,
                "pred": pred,
                "prob": prob if pred == 1 else 1-prob
            }
    
    def read_data(self):
        df = pd.read_csv(self.data_path)

        if self.num_samples > 0:
            df = df.iloc[:self.num_samples]

        data = []
        for idx, row in df.iterrows():
            data.append({
                "id": idx,
                "image_path": row["file_path"],
                "label": int(row["label"])
            })

        return data
    
    def get_fewshot_images(self):
        images_rootdir = "/work/projects/polyullm/houzht/PrePath/api_test/fewshot_images"
        images = ["672304.png", "683520.png", "671393.png", "691683.jpg"]
        fewshot_images_prompt = []
        for image in images:
            base64_image, mime_name = encode_image(os.path.join(images_rootdir, image))
            if not isinstance(base64_image, List):
                base64_image = [base64_image]
            for bi in base64_image:
                fewshot_images_prompt.append({"type": "image_url", "image_url": f"data:{mime_name};base64,{bi}"})
        return fewshot_images_prompt
    
    @staticmethod
    def read_jsonl(path):
        if not os.path.exists(path):
            return {}
        
        data = {}
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                idx = obj["id"]
                data[idx] = obj
        return data

    @staticmethod
    def append_jsonl(path, obj):
        with open(path, "a", encoding="utf8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def few_shot_prompt():
        pass

    def get_prompt(self, slide_id):
        if self.use_dnn:
            pred, prob = LABEL_MAP[int(self.dnn_results[slide_id]["pred"])], self.dnn_results[slide_id]["prob"]
            # prob = random.uniform(0.8, 1.0)  # random.random()
        fewshot_prompt = """

将会提供多张图像，请按顺序理解：

第 1 张图：鼻咽癌示例（用于参考，不需要判断）
第 2 张图：鼻咽癌示例（用于参考，不需要判断）
第 3 张图：非鼻咽癌示例（用于参考，不需要判断）
第 4 张图：非鼻咽癌示例（用于参考，不需要判断）
剩下的图像：可能是一个图像的多个 tiles, 需要你判断是否为鼻咽癌。

""" if self.use_fewshot else ""

        dnn_prompt = f"""
        
我们已有的判断结果是 {pred}，置信度为 {round(prob, 4)*100}%。该模型判断具有极高可靠性，除非有明确证据，否则谨慎改变原来的判断。请你基于已有的判断结果，再次进行判断

""" if self.use_dnn else ""
        
        prompt = f"""你是一个鼻咽癌病理图像专家。你的任务是分析提供给你的病理图像，并判断该图像是否为鼻咽癌。
{fewshot_prompt}
{dnn_prompt}
请严格按照以下格式回答：
<class>类别</class>
<reasoning>理由</reasoning>

类别说明：
请严格输出 "非鼻咽癌" 或 "鼻咽癌"，不可输出其他值或文字。

理由说明：
请详细说明你是如何从图像中得出结论的，可以参考细胞形态、核形态、组织结构、染色特点等典型病理特征。  
请注意：
- 只能基于图像信息判断，不要凭空想象。  
- 绿色曲线标注仅代表有效染色区域，不代表病变区域，请勿将其作为判断依据。"""
        return prompt

    def process_messages(self):
        existing = self.read_jsonl(self.output_path)
        print(f"[Resume] Loaded {len(existing)} previous results from {self.output_path}")

        new_messages = []
        for data in tqdm(self.data, desc="Split Image ..."):
            self.idx_list.append(data["id"])
            if data["id"] in existing:
                new_messages.append({})
                continue
            image_path = data["image_path"]
            slide_id = os.path.basename(os.path.dirname(image_path))
            prompt = self.get_prompt(slide_id)
            base64_image, mime_name = encode_image(image_path, split=False)
            fewshot_images = self.fewshot_images[:]
            if not isinstance(base64_image, List):
                base64_image = [base64_image]
            for bi in base64_image:
                fewshot_images.extend([{"type": "image_url", "image_url": f"data:{mime_name};base64,{bi}"}])
            fewshot_images.extend([{"type": "text", "text": prompt}])
            new_messages.append({"messages": [{"role": "user", "content": fewshot_images}], "label": data["label"], "file_path": image_path, "tiles": len(base64_image), "prompt": prompt})

        return new_messages

    def process_single_messages(self, messages):
        # print(messages)
        # import pdb; pdb.set_trace()

        messages = messages["messages"]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, _ = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt,
            "image_data": image_inputs
        }
        return llm_inputs

    def generate_output(self, messages):
        llm_inputs = self.process_single_messages(messages)
        outputs = self.llm.generate(prompt=llm_inputs["prompt"], image_data=llm_inputs["image_data"], sampling_params=self.sampling_params)
        del llm_inputs
        return outputs["text"]

        
    def generate_outputs(self, output_path: str = ""):
        existing = self.read_jsonl(output_path)
        print(f"[Resume] Loaded {len(existing)} previous results from {output_path}")

        res = []
        for idx, messages in tqdm(zip(self.idx_list, self.messages_list), total=len(self.messages_list), desc="Generate outputs"):
            if idx in existing:
                res.append(existing[idx])
                continue
            result = self.generate_output(messages)
            answer = extract(result, "answer", hard=True)
            pred = extract(answer, "class", hard=False, number=True)
            result = {"id": idx, "prompt": messages["prompt"], "response": result, "label": messages["label"], "correct": int(pred) == messages["label"] if pred != "" else False, "file_path": messages["file_path"], "tiles": messages["tiles"]}
            self.append_jsonl(output_path, result)
            res.append(result)
        
        return res
    
    async def async_generate_output(self, messages):
        llm_inputs = self.process_single_messages(messages)
        outputs = await self.llm.async_generate(prompt=llm_inputs["prompt"], image_data=llm_inputs["image_data"], sampling_params=self.sampling_params)
        del llm_inputs
        return outputs["text"]

    async def async_generate_outputs(self, output_path: str = ""):
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results = [None] * len(self.messages_list)

        async def _worker(idx, payload):
            async with semaphore:
                results[idx] = await self.async_generate_output(payload)
                answer = extract(results[idx], "answer", hard=True)
                pred = extract(answer, "class", hard=False, number=True)
                result = {"id": idx, "prompt": payload["prompt"], "response": results[idx], "label": payload["label"], "correct": int(pred) == payload["label"] if pred != "" else False, "file_path": payload["file_path"], "tiles": payload["tiles"]}
                self.append_jsonl(output_path, result)
        
        tasks = [asyncio.create_task(_worker(i, m)) for i, m in enumerate(self.messages_list)]

        # show progress as tasks complete
        for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async generate outputs"):
            await _

        return results


class Qwen3_5:
    def __init__(self, args):
        print("tp size:", int(os.environ.get("TENSOR_PARALLEL_SIZE", 1)))
        print("dp size:", int(os.environ.get("DATA_PARALLEL_SIZE", 1)))
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.config = AutoConfig.from_pretrained(args.model_path)
        self.context_length = self.tokenizer.model_max_length
        self.llm = Engine(
            model_path=args.model_path,
            enable_multimodal=True,
            mem_fraction_static=0.8,
            tp_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", 1)),
            dp_size=int(os.environ.get("DATA_PARALLEL_SIZE", 1)),
            attention_backend="fa3",
            context_length=self.context_length,
            # enable_torch_compile=False,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            enable_deterministic_inference=True,
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path, padding_side="right")
        self.max_concurrency = args.model_max_concurrency
        self.split = args.split.lower() == "true"
        self.enable_thinking = True if args.enable_thinking.lower() == "true" else False
        self.default_system_prompt = args.default_system_prompt

        self.sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }

        print(f"default_system_prompt: {args.default_system_prompt}, {self.sampling_params}")

        self.num_samples = args.num_samples
        self.use_fewshot = args.use_fewshot.lower() == "true"
        self.use_dnn = args.dnn_results != ""
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.data = self.read_data()
        self.idx_list = []

        self.dnn_results = {}

        if args.dnn_results:
            self.read_dnn_results(args.dnn_results)

        self.fewshot_images = self.get_fewshot_images() if self.use_fewshot else []
        self.messages_list = self.process_messages()

    def read_dnn_results(self, dnn_results):
        df = pd.read_csv(dnn_results, encoding="utf-8")

        for _, row in df.iterrows():
            slide_path = row["slide_path"]
            label = int(row["label"])
            pred = int(row["pred"])
            prob = float(row["prob"])

            slide_name = os.path.basename(slide_path)
            slide_id = os.path.splitext(slide_name)[0]

            self.dnn_results[slide_id] = {
                "label": label,
                "pred": pred,
                "prob": prob if pred == 1 else 1-prob
            }
    
    def read_data(self):
        df = pd.read_csv(self.data_path)

        if self.num_samples > 0:
            df = df.iloc[:self.num_samples]

        data = []
        for idx, row in df.iterrows():
            data.append({
                "id": idx,
                "image_path": row["file_path"],
                "label": int(row["label"])
            })

        return data
    
    def get_fewshot_images(self):
        images_rootdir = "/work/projects/polyullm/houzht/PrePath/api_test/fewshot_images"
        images = ["672304.png", "683520.png", "671393.png", "691683.jpg"]
        fewshot_images_prompt = []
        for image in images:
            base64_image, mime_name = encode_image(os.path.join(images_rootdir, image))
            if not isinstance(base64_image, List):
                base64_image = [base64_image]
            for bi in base64_image:
                fewshot_images_prompt.append({"type": "image_url", "image_url": f"data:{mime_name};base64,{bi}"})
        return fewshot_images_prompt
    
    @staticmethod
    def read_jsonl(path):
        if not os.path.exists(path):
            return {}
        
        data = {}
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                idx = obj["id"]
                data[idx] = obj
        return data

    @staticmethod
    def append_jsonl(path, obj):
        with open(path, "a", encoding="utf8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def few_shot_prompt():
        pass

    def get_prompt(self, slide_id):
        if self.use_dnn:
            pred, prob = LABEL_MAP[int(self.dnn_results[slide_id]["pred"])], self.dnn_results[slide_id]["prob"]
            # prob = random.uniform(0.8, 1.0)  # random.random()
        fewshot_prompt = """

将会提供多张图像，请按顺序理解：

第 1 张图：鼻咽癌示例（用于参考，不需要判断）
第 2 张图：鼻咽癌示例（用于参考，不需要判断）
第 3 张图：非鼻咽癌示例（用于参考，不需要判断）
第 4 张图：非鼻咽癌示例（用于参考，不需要判断）
剩下的图像：可能是一个图像的多个 tiles, 需要你判断是否为鼻咽癌。

""" if self.use_fewshot else ""

        dnn_prompt = f"""
        
我们已有的判断结果是 {pred}，置信度为 {round(prob, 4)*100}%。该模型判断具有极高可靠性，除非有明确证据，否则谨慎改变原来的判断。请你基于已有的判断结果，再次进行判断

""" if self.use_dnn else ""
        
        prompt = f"""你是一个鼻咽癌病理图像专家。你的任务是分析提供给你的病理图像，并判断该图像是否为鼻咽癌。
{fewshot_prompt}
{dnn_prompt}
请严格按照以下格式回答：
<class>类别</class>
<reasoning>理由</reasoning>

类别说明：
请严格输出 "非鼻咽癌" 或 "鼻咽癌"，不可输出其他值或文字。

理由说明：
请详细说明你是如何从图像中得出结论的，可以参考细胞形态、核形态、组织结构、染色特点等典型病理特征。  
请注意：
- 只能基于图像信息判断，不要凭空想象。  
- 绿色曲线标注仅代表有效染色区域，不代表病变区域，请勿将其作为判断依据。"""
        return prompt

    def process_messages(self):
        existing = self.read_jsonl(self.output_path)
        print(f"[Resume] Loaded {len(existing)} previous results from {self.output_path}")

        new_messages = []
        for data in tqdm(self.data, desc="Split Image ..."):
            self.idx_list.append(data["id"])
            if data["id"] in existing:
                new_messages.append({})
                continue
            image_path = data["image_path"]
            slide_id = os.path.basename(os.path.dirname(image_path))
            prompt = self.get_prompt(slide_id)
            base64_image, mime_name = encode_image(image_path, split=False)
            fewshot_images = self.fewshot_images[:]
            if not isinstance(base64_image, List):
                base64_image = [base64_image]
            for bi in base64_image:
                fewshot_images.extend([{"type": "image", "image": f"data:{mime_name};base64,{bi}"}])
            fewshot_images.extend([{"type": "text", "text": prompt}])
            new_messages.append({"messages": [{"role": "user", "content": fewshot_images}], "label": data["label"], "file_path": image_path, "tiles": len(base64_image), "prompt": prompt})

        return new_messages

    def process_single_messages(self, messages):
        # print(messages)
        # import pdb; pdb.set_trace()

        messages = messages["messages"]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        image_inputs, _ = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt,
            "image_data": image_inputs
        }

        return llm_inputs

    def generate_output(self, messages):
        llm_inputs = self.process_single_messages(messages)
        outputs = self.llm.generate(prompt=llm_inputs["prompt"], image_data=llm_inputs["image_data"], sampling_params=self.sampling_params)
        del llm_inputs
        return outputs["text"].split("</think>")[-1]

        
    def generate_outputs(self, output_path: str = ""):
        existing = self.read_jsonl(output_path)
        print(f"[Resume] Loaded {len(existing)} previous results from {output_path}")

        res = []
        for idx, messages in tqdm(zip(self.idx_list, self.messages_list), total=len(self.messages_list), desc="Generate outputs"):
            if idx in existing:
                res.append(existing[idx])
                continue
            result = self.generate_output(messages)
            answer = extract(result, "answer", hard=True)
            pred = extract(answer, "class", hard=False, number=True)
            result = {"id": idx, "prompt": messages["prompt"], "response": result, "label": messages["label"], "correct": int(pred) == messages["label"] if pred != "" else False, "file_path": messages["file_path"], "tiles": messages["tiles"]}
            self.append_jsonl(output_path, result)
            res.append(result)
        
        return res
    
    async def async_generate_output(self, messages):
        llm_inputs = self.process_single_messages(messages)
        outputs = await self.llm.async_generate(prompt=llm_inputs["prompt"], image_data=llm_inputs["image_data"], sampling_params=self.sampling_params)
        del llm_inputs
        return outputs["text"].split("</think>")[-1]

    async def async_generate_outputs(self, output_path: str = ""):
        existing = self.read_jsonl(output_path)
        print(f"[Resume] Loaded {len(existing)} previous results from {output_path}")

        semaphore = asyncio.Semaphore(self.max_concurrency)
        results = [None] * len(self.messages_list)

        async def _worker(idx, sample_id, payload):
            if sample_id in existing:
                results[idx] = existing[sample_id]
                return 

            async with semaphore:
                results[idx] = await self.async_generate_output(payload)
                # write to jsonl
                answer = extract(results[idx], "answer", hard=True)
                pred = extract(answer, "class", hard=False, number=True)
                results[idx] = {"id": sample_id, "prompt": payload["prompt"], "response": results[idx], "label": payload["label"], "correct": int(pred) == payload["label"] if pred != "" else False, "file_path": payload["file_path"], "tiles": payload["tiles"]}
                self.append_jsonl(output_path, results[idx])
                
        
        tasks = [asyncio.create_task(_worker(i, sample_id, m)) for i, (sample_id, m) in enumerate(zip(self.idx_list, self.messages_list))]

        # show progress as tasks complete
        for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async generate outputs"):
            await _
        
        print(len(results))
        return results


def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type, hard = True, number: bool = False, label_map: Dict = None):
    label_map = {"非鼻咽癌": 0, "鼻咽癌": 1}
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            if number:
                target_str = label_map.get(target_str, -1)
            return target_str
        elif hard:
            if number:
                target_str = label_map.get(text, text)
            return text
        else:
            return ""
    else:
        return ""


def evaluation(results):
    y_pred = [] 
    y_true = []
    for result in results:
        result = json.loads(result) if isinstance(result, str) else result
        answer = extract(result["response"], "answer", hard=True)
        pred = extract(answer, "class", hard=False, number=True)
        label = result["label"]
        if "677962.jpg" in result["file_path"] or "680564.jpg" in result["file_path"] or "538182.jpg" in result["file_path"]:
            continue

        y_pred.append(int(pred) if pred != "" else -1)
        y_true.append(label)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    mask = y_pred != -1
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    valid_count = mask.sum()
    print(f"Valid predictions  : {valid_count}")

    if len(y_pred) == 0:
        print("No valid predictions.")
        return

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # == sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp + 1e-8)  # division 0
    sensitivity = recall  # same as recall
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = np.nan

    print("\n===== Slide-level Metrics =====")
    print(f"AUC          : {auc:.4f}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall (Sens): {recall:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Confusion Mat:\n{cm}")


def main():
    args = get_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["TENSOR_PARALLEL_SIZE"] = args.tensor_parallel_size
    os.environ["DATA_PARALLEL_SIZE"] = args.data_parallel_size
    if int(args.tensor_parallel_size) > 1 or int(args.data_parallel_size) > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.model_path:
        if args.model_type == "qwen3":
            api_model = Qwen3VL(args)
        elif args.model_type == "qwen3_5":
            api_model = Qwen3_5(args)
        else:
            raise ValueError(f"{args.model_type} is not supported.")
    else:
        api_model = API_MODEL(args)

    if args.use_async.lower() in ["true", "True"]:
        if args.model_path:
            results = asyncio.run(api_model.async_generate_outputs(args.output_path))
        else:
            results = asyncio.run(api_model.async_generate_outputs(args.output_path))
    else:
        results = api_model.generate_outputs(args.output_path)

    evaluation(results)


if __name__ == "__main__":
    main()
