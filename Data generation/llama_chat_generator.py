from typing import List, Optional
import fire
from llama import Llama, Dialog

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """ 
    """
    torchrun --nproc_per_node 1 example_chat_completion.py 
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
    """
    """
    torchrun --nproc_per_node 1 ../Dissertation-main/Llama/llama_chat.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model  --max_seq_len 512 --max_batch_size 8  
    """
    generator = Llama.build(
        ckpt_dir='llama-2-7b-chat/',
        tokenizer_path='tokenizer.model',
        max_seq_len=1024,
        max_batch_size=8,
    )
    
    system_contents = {
        'A': "You are a human text chatting your coworker and trying to deceive your coworker about buying a new gadget. send short messages",
        'B': "You are a human. response to the text messages that your coworker is sending you. send short messages"
        }

    def prompt_generator(person, messages, system):
        #print(messages)
        prompt = [
            {
                "role": "system",
                "content": system
            }
        ]
         # llama expects the first role after system to be user not assitant and then (u/a/u/a/u...) and the last role always shoul be user
        # so the number of turns betweeen user and assistant should be always odd starting and ending by user role
        # so we check and if the number of turns is even we add the first probmt
        # but is this a good way to hndle roles!?
        if ~len(messages) %2: # even
            messages.insert(0, {'A' : 'start by asking how they are doing'})  
        else:
            if len(messages) > 1:
                messages.pop(0)
        
        for i, j  in enumerate(messages):
            for p, m in j.items():   
                if i%2 : #odd
                    role = 'assistant'
                else:
                    role = 'user'
            prompt.append({"role" : role , "content" : m}) 
        return prompt

    def get_response(person, messages):
        prompt = prompt_generator(person, messages, system_contents[person])
        dialogs: List[Dialog] = [prompt]
        #print(dialogs)
        response =  generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        #print(response)
        return response[0]['generation']['content'] 
     
    messages = []
    response_of_A =  get_response('A',[{'A' : 'start by asking how they are doing'}])  

    for _ in range(5): 
        print("A:" , response_of_A)
        print("\n==================================\n")
        messages.append({'B' : response_of_A})
        response_of_B = get_response('B', messages)  
        
        messages.append({'A' : response_of_B})
        response_of_A = get_response('A', messages)
            
        print("B:" , response_of_B)
        print("\n==================================\n")
        

if __name__ == "__main__":
    fire.Fire(main)
