class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

        # NEW: store user info (like name)
        self.user_profile = {
            "name": None
        }

    # -----------------------------------
    # ADD TURN
    # -----------------------------------
    def add_turn(self, user, bot):
        self.history.append({"user": user, "bot": bot})

        if len(self.history) > self.max_history:
            self.history.pop(0)

        # NEW: auto-detect name
        self.extract_user_info(user)

    # -----------------------------------
    # NEW: EXTRACT USER INFO (SMART)
    # -----------------------------------
    def extract_user_info(self, user_input):

        user_input = user_input.lower()

        # detect name
        if "my name is" in user_input:
            name = user_input.split("my name is")[-1].strip().split()[0]
            self.user_profile["name"] = name.capitalize()

        elif "i am" in user_input and len(user_input.split()) <= 5:
            # simple "I am Prosper"
            name = user_input.split("i am")[-1].strip().split()[0]
            if len(name) < 15:
                self.user_profile["name"] = name.capitalize()

    # -----------------------------------
    # GET CONTEXT (CLEAN FORMAT)
    # -----------------------------------
    def get_context(self):

        context = ""

        for turn in self.history:
            context += f"{turn['user']}\n{turn['bot']}\n"

        return context.strip()

    # -----------------------------------
    # NEW: GET USER NAME
    # -----------------------------------
    def get_user_name(self):
        return self.user_profile.get("name")

    # -----------------------------------
    # NEW: RESET MEMORY (OPTIONAL)
    # -----------------------------------
    def reset(self):
        self.history = []
        self.user_profile = {"name": None}