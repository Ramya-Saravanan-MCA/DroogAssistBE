#input token ccst - 0.20 and output token cost-0.80 per million

def compute_llm_cost(input_tokens, output_tokens, input_cost=0.20, output_cost=0.80):
    # Costs per million tokens, so divide tokens by 1_000_000
    return round(
        ((input_tokens or 0) * (input_cost / 1_000_000)) +
        ((output_tokens or 0) * (output_cost / 1_000_000)),
        6
    )