import logging

class SingletonToken:
    """
    Implements a singleton token.

    This class is used to implement a singleton token which can be set and retrieved using the 
    set_token() and get_token() class methods respectively. The token is stored in a private class 
    variable and can be accessed across different instances of the class.
    """
    __token = None

    @classmethod
    def set_token(cls, token):
        cls.__token = token

    @classmethod
    def get_token(cls):
        return cls.__token

def get_final_thought_and_answer(output):
    # Split the output into two parts based on the first occurrence of "Final Answer:"
    parts = output.split("Final Answer:", 1)

    # If the output was successfully split into two parts
    if len(parts) == 2:
        # The final answer is the second part, stripped of leading and trailing whitespace
        final_answer = parts[1].strip()

        # Then split the first part of the output based on the last occurrence of "Thought:"
        thoughts = parts[0].rsplit("Thought:", 1)

        # If the first part was successfully split into two parts
        if len(thoughts) == 2:
            # The final thought is the second part, stripped of leading and trailing whitespace
            final_thought = thoughts[1].strip()
        else:
            # If the first part could not be split, there was no final thought
            final_thought = "No final thought provided."
    else:
        # If the output could not be split, there was no final answer
        final_answer = "No final answer provided."
        final_thought = "No final thought provided."

    return final_thought, final_answer




# def get_final_answer(output):
#     # Split the output into two parts based on the first occurrence of "Final Answer:"
#     parts = output.split("Final Answer:", 1)

#     # If the output was successfully split into two parts
#     if len(parts) == 2:
#         # The final answer is the second part, stripped of leading and trailing whitespace
#         final_answer = parts[1].strip()
#     else:
#         # If the output could not be split, there was no final answer
#         final_answer = "No final answer provided."

#     return final_answer