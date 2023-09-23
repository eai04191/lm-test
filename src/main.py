import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class ChatBot:
    def __init__(self, model_name: str, timeout: float = 10.0):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).half().to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.timeout = timeout
        self.prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: <prompt> "
            "ASSISTANT:"
        )

    def respond(self, user_input: str):
        text = self.prompt.replace("<prompt>", user_input)
        token_ids = self.tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        )
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=self.timeout,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def generate(input):
            try:
                self.model.generate(
                    input,
                    streamer=streamer,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=4096,
                    temperature=0.9,
                    top_p=0.8,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.25,
                )
            except Exception as e:
                print(f"\nError during generation: {e}")

        thread = Thread(target=generate, args=(token_ids.to(self.model.device),), daemon=True)
        thread.start()

        print("ASSISTANT: ", end="")
        for new_text in streamer:
            print(new_text, end="")
        print("\n")


def main():
    chat_bot = ChatBot("Xwin-LM/Xwin-LM-7B-V0.1")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            chat_bot.respond(user_input)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
