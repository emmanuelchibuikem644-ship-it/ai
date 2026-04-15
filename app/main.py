from inference.response_generator import ResponseGenerator


def main():
    print(" Chatbot ready. Type 'exit' to quit.\n")

    bot = ResponseGenerator()

    while True:
        try:
            user_input = input("You: ")

            if not user_input.strip():
                continue  # skip empty input

            if user_input.lower() in ["exit", "quit"]:
                print("Bot: Take care of yourself. I'm here whenever you need support.")
                break

            result = bot.generate(user_input)

            print(f"Detected Emotion: {result['emotion']}")
            print(f"Bot: {result['response']}\n")

        except KeyboardInterrupt:
            print("\nBot: Goodbye ")
            break

        except Exception as e:
            print(f"\n Error: {e}\n")


if __name__ == "__main__":
    main()