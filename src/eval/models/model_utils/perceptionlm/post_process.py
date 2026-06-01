def post_process_response(response: str) -> str:
    return response.strip().split(".")[0].strip()
