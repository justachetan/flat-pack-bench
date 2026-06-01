from openai import OpenAI
import os.path as osp
import pyrootutils

root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
dummy_image_uri = "file://" + osp.join(
    root,
    "src",
    "eval",
    "models",
    "dummy_data",
    "9fa8c6168c5f381e1e28551a65930d9b183ecbfb5f14ec97b8d37ced4f275eaa.jpg",
)

with open(osp.join(root, "src", "tva", "media", "prompts", "qwen25_vl_vllm_online.jinja"), "r") as f:
    chat_template = f.read()

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is happening in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    # Option A: HTTP(S)
                    # "url": "https://example.com/image.jpg"

                    # Option B: base64 data URI
                    # "url": "data:image/jpeg;base64,...."

                    # Option C: file URI (see caveat below)
                    "url": dummy_image_uri
                },
            },
        ],
    }],
    max_tokens=65536,
    extra_body={
        "chat_template": chat_template,
        "chat_template_kwargs": {"add_vision_id": True},
        "add_generation_prompt": True,
    },
    temperature=0.1,
)
# import ipdb; ipdb.set_trace()
print(resp.choices[0].message.content)
