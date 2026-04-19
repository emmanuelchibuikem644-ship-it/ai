class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

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

        self.extract_user_info(user)

    # -----------------------------------
    # EXTRACT USER INFO
    # -----------------------------------
    def extract_user_info(self, user_input):

        text = user_input.lower().strip()

        if "my name is" in text:
            name = text.split("my name is")[-1].strip().split()[0]
            if self._valid_name(name):
                self.user_profile["name"] = name.capitalize()

        elif text.startswith("i am "):
            name = text.replace("i am", "").strip().split()[0]
            if self._valid_name(name):
                self.user_profile["name"] = name.capitalize()

        elif text.startswith("i'm "):
            name = text.replace("i'm", "").strip().split()[0]
            if self._valid_name(name):
                self.user_profile["name"] = name.capitalize()

    # -----------------------------------
    # VALIDATE NAME
    # -----------------------------------
    def _valid_name(self, name):

        invalid_words = [
            "sad", "tired", "happy", "angry",
            "fine", "okay", "good", "bad"
        ]

        return name.isalpha() and name not in invalid_words and len(name) < 15

    # -----------------------------------
    # GET CONTEXT
    # -----------------------------------
    def get_context(self):

        context = ""

        for turn in self.history:
            context += f"User: {turn['user']}\nFriend: {turn['bot']}\n"

        return context.strip()

    # -----------------------------------
    # FIX: GET HISTORY (MISSING METHOD)
    # -----------------------------------
    def get_history(self):
        return self.history

    # -----------------------------------
    # GET USER NAME
    # -----------------------------------
    def get_user_name(self):
        return self.user_profile.get("name")

    # -----------------------------------
    # PERSONALIZE TEXT
    # -----------------------------------
    def personalize(self, text):

        name = self.get_user_name()

        if name and name.lower() not in text.lower():
            return f"{name}, {text}"

        return text

    # -----------------------------------
    # RESET MEMORY
    # -----------------------------------
    def reset(self):
        self.history = []
        self.user_profile = {"name": None}