'''Note to self: possible that we'll be able to share LM heads '''
import torch
from liger_kernel.transformers import AutoLigerKernelForCausalLM, LigerFusedLinearJSD

'''Want something like the below '''
async def get_teacher_logprobs(client, sequence, idx, semaphore):
    async with semaphore:
        response = await client.completions.create(
            model="allenai/Olmo-3-7B-Instruct",
            prompt=sequence,  # full prompt + completion concatenated
            max_tokens=1,
            extra_body={"prompt_logprobs": 128}  # top-128 for KLD
        )
        return {
            'idx': idx,
            'prompt_logprobs': response.choices[0].prompt_logprobs
        }



kld_loss = LigerFusedLinearJSD(jsd_beta=1) #JSD with beta = 0 is forward kl, beta = 1 is reverse kl. might be fun to try with JSD in the future though
