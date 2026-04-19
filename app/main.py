from inference.response_generator import ResponseGenerator


def main():
    print(" Chatbot ready. Type 'exit' to quit.\n")

    bot = ResponseGenerator()

    # ✅ ADDED: simple session memory (DO NOT change structure)
    chat_history = []

    def add_to_history(role, message):
        chat_history.append(f"{role}: {message}")
        if len(chat_history) > 8:  # keep last 4 exchanges
            chat_history.pop(0)

    while True:
        try:
            user_input = input("You: ")

            if not user_input.strip():
                continue  # skip empty input

            if user_input.lower() in ["exit", "quit"]:
                print("Bot: Take care of yourself. I'm here whenever you need support.")
                break

            # ✅ ADD USER MESSAGE TO MEMORY
            add_to_history("User", user_input)

            # 🔥 SEND CONTEXT TO MODEL (IMPORTANT IMPROVEMENT)
            context = "\n".join(chat_history)

            # If your ResponseGenerator supports context, pass it
            try:
                result = bot.generate(user_input, context=context)
            except TypeError:
                # fallback if your function doesn't accept context yet
                result = bot.generate(user_input)

            #  ADD BOT RESPONSE TO MEMORY
            add_to_history("Bot", result["response"])

            print(f"Detected Emotion: {result['emotion']}")
            print(f"Bot: {result['response']}\n")

        except KeyboardInterrupt:
            print("\nBot: Goodbye ")
            break

        except Exception as e:
            print(f"\n Error: {e}\n")


if __name__ == "__main__":
    main()