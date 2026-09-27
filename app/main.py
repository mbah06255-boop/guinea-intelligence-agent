from agent import Agent


def main():
    agent = Agent()
    print("Agent IA prêt (web_search, calculator, get_datetime). Tape 'quit' pour arrêter.\n")

    while True:
        user_input = input("Toi : ")
        if user_input.lower() in ("quit", "exit"):
            break

        reply = agent.ask(user_input)
        print(f"Agent : {reply}\n")


if __name__ == "__main__":
    main()