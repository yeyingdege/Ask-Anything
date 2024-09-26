import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from torchvision import transforms

from transformers import StoppingCriteria, StoppingCriteriaList
from peft import get_peft_model, LoraConfig, TaskType

from utils.easydict import EasyDict
from utils.config import Config
from utils.decord_func import decord_video_given_start_end_seconds
from utils.eval_utils import parse_choice, TypeAccuracy
from dataset.hd_utils import HD_transform_padding, HD_transform_no_padding
from models.videochat_mistra.videochat2_it_hd_mistral import VideoChat2_it_hd_mistral



QUESTION_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
                  'qa4_step','qa5_task', 'qa6_precedingStep', 'qa7_bestPrecedingStep',
                  'qa8_toolNextStep', 'qa9_bestInitial','qa10_bestFinal', 'qa11_domain']


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret

def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda"),
        torch.tensor([29871, 2]).to("cuda")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
    # output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, start_secs=-1, end_secs=-1,
               return_msg=False, resolution=224, hd_num=6, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    frame_indices = decord_video_given_start_end_seconds(video_path, 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    # print(frames.shape)
    
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
    
def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

    # print(f"n_position: {n_position}")
    # print(f"pre_n_position: {pre_n_position}")

    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table


def load_ckpts(cfg):
    # load stage2 model
    cfg.model.vision_encoder.num_frames = 4
    model = VideoChat2_it_hd_mistral(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()

    # add lora to run stage3 model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
    )
    model.mistral_model = get_peft_model(model.mistral_model, peft_config)

    state_dict = torch.load(cfg.stage4_model, cfg.device)
    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)
    return model


def inference_video(model, text, vid_path, 
                    num_frame=16, start_secs=-1, end_secs=-1,
                    resolution=224, hd_num=12, padding=False):
    vid, msg = load_video(
        vid_path, num_segments=num_frame, return_msg=True, 
        start_secs=start_secs, end_secs=end_secs,
        resolution=resolution, hd_num=hd_num, padding=padding
    )
    # print('start_secs', start_secs, 'end_secs', end_secs, msg)

    # The model expects inputs of shape: T x C x H x W
    T_, C, H, W = vid.shape
    video = vid.reshape(1, T_, C, H, W).to("cuda")

    img_list = []
    with torch.no_grad():
        image_emb, _, _ = model.encode_img(video, "Watch the video and answer the question.")
    img_list.append(image_emb[0])

    chat = EasyDict({
        "system": "",
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
    # chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg} [/INST]"])
    ask(text, chat)

    llm_message = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=512)[0]
    return llm_message



def main(args):
    cfg = Config.from_file(args.config_file)

    model = load_ckpts(cfg)
    model = model.eval()
    ## input settings
    num_frame = args.num_video_frames
    # resolution = 384
    resolution = 224
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
    model.vision_encoder.encoder.pos_embed = new_pos_emb


    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Overall Accuracy for All Questions
    global_acc = TypeAccuracy("Global")
    qa_acc = []
    for t in range(len(QUESTION_TYPES)):
        qa_acc.append(TypeAccuracy(f"qa{t+1}_"))


    total = 0
    results = {}
    for line in tqdm(annotations, total=len(annotations)):
        # Q-A Pair
        idx = line["qid"]
        quest_type = line["quest_type"]
        conversations = line["conversations"]
        qs = conversations[0]["value"]
        gt_answers = conversations[1]["value"]
        results[idx] = {"qid": idx, "quest_type": quest_type, 
                        "qs": qs, "gt": gt_answers,
                        "task_label": line["task_label"], 
                        "step_label": line["step_label"]}
        qs = qs.replace("<video>\n", "")
        qs = qs.replace("<image>\n", "")
        vid_path = os.path.join(args.image_folder, line["video"])
        if "start_secs" in line:
            start_secs = line['start_secs']
            end_secs = line['end_secs']
        else:
            start_secs = -1
            end_secs = -1
        response = inference_video(model, text=qs, vid_path=vid_path, num_frame=num_frame, 
                                   start_secs=start_secs, end_secs=end_secs)

        total += 1
        answer_id = parse_choice(response, line["all_choices"], line["index2ans"])
        results[idx]["response"] = response
        results[idx]["parser"] = answer_id
        # print("qid {}:\n{}".format(idx, qs))
        # print("AI: {}\nParser: {}\nGT: {}\n".format(response, answer_id, gt_answers))

        global_acc.update(gt_answers, answer_id)
        for t in range(len(QUESTION_TYPES)):
            if f"qa{t+1}_" in quest_type:
                qa_acc[t].update(gt_answers, answer_id)

        # print each type accuracy
        print("-----"*5)
        acc_list = []
        for t in range(len(QUESTION_TYPES)):
            qa_acc[t].print_accuracy()
            acc_list.append(qa_acc[t].get_accuracy())
        global_acc.print_accuracy()
        print("-----"*5)
        avg_acc = sum(acc_list) / len(acc_list)
        print("Average Acc over Type: {:.4f}".format(avg_acc))

    # save all results
    print("save to {}".format(args.answers_file))
    with open(args.answers_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="configs/config_mistral_hd.json")
    parser.add_argument("--stage4_model", type=str, default="ckpt/videochat2_hd_mistral_7b_stage4.pth")
    parser.add_argument("--image-folder", type=str, default="data/COIN/videos")
    parser.add_argument("--question-file", type=str, default="data/testing_vqa.json")
    parser.add_argument("--answers-file", type=str, default="data/answers_videochat2_f16.json")
    # parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_video_frames", type=int, default=16)
    args = parser.parse_args()
    main(args)

