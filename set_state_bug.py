import openvino
import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

device = "GPU"
ov_config = {}

if "GPU" in device:
    ov_config["KV_CACHE_PRECISION"] = "undefined"

model_id = "Qwen/Qwen3-1.7B"
model = OVModelForCausalLM.from_pretrained(model_id, export=True, device=device, ov_config=ov_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

system_message = {"role": "system", "content": "Your name is OpenVINO."}

message1 = [system_message, {"role": "user", "content": "What's your name?"}]

message2 = [system_message, {"role": "user", "content": "Please tell me your name."}]

print("Message 1 ...")

inputs1 = tokenizer.apply_chat_template(message1, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(
    "cpu"
)
input1_length = inputs1["input_ids"].shape[1]
outputs1 = model.generate(**inputs1, do_sample=False, max_new_tokens=1024)
response1 = tokenizer.decode(outputs1[0, input1_length:], skip_special_tokens=True)

print(response1)

infer_req = model.request

key_caches = {}
value_caches = {}
cached_tokens = outputs1[0, :-1]
cached_length = len(cached_tokens)

# print(f"cached_tokens: {cached_tokens}")

print(f"Storing prefix cache... caching {cached_length} tokens")

for var_state in infer_req.query_state():
    if "past_key_values." in var_state.name:
        # print(f"get state {var_state.name} {var_state.state}")
        name_parts = var_state.name.split(".")

        layer = int(name_parts[1])

        if name_parts[4] == "key":
            key_caches[layer] = torch.from_numpy(var_state.state.data)
        elif name_parts[4] == "value":
            value_caches[layer] = torch.from_numpy(var_state.state.data)
        else:
            continue

cache = DynamicCache()

for layer in key_caches:
    cache.update(key_caches[layer], value_caches[layer], layer)

print("Message 2 ...")

inputs2 = tokenizer.apply_chat_template(message2, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(
    "cpu"
)
input2_length = inputs2["input_ids"].shape[1]

match_position = -1

while True:
    next_pos = match_position + 1

    if next_pos >= input2_length or next_pos >= cached_length:
        break

    if inputs2["input_ids"][0][next_pos] != cached_tokens[next_pos]:
        break

    match_position = next_pos

print(f"Applying prefix cache... prefix ending at {match_position}")

cache.crop(match_position + 1)

infer_req.reset_state()

for var_state in infer_req.query_state():
    if "past_key_values." in var_state.name:
        name_parts = var_state.name.split(".")

        layer = int(name_parts[1])

        if name_parts[4] == "key":
            var_state.state = openvino.Tensor(cache[layer][0].numpy())
        elif name_parts[4] == "value":
            var_state.state = openvino.Tensor(cache[layer][1].numpy())
        else:
            continue

        # print(f"set state {var_state.name} {var_state.state}")

model._past_length = cache.get_seq_length()

inputs2["position_ids"] = torch.Tensor([list(range(match_position + 1, len(inputs2["input_ids"][0])))])

outputs2 = model.generate(**inputs2, do_sample=False, max_new_tokens=1024, past_key_values=cache.to_legacy_cache())
input2_length = inputs2["input_ids"].shape[1]
response2 = tokenizer.decode(outputs2[0, input2_length:], skip_special_tokens=True)
print(response2)
