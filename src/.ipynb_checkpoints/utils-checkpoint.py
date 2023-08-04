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


def get_final_answer(output):
    # Split the output into two parts based on the first occurrence of "Final Answer:"
    parts = output.split("Final Answer:", 1)

    # If the output was successfully split into two parts
    if len(parts) == 2:
        # The final answer is the second part, stripped of leading and trailing whitespace
        final_answer = parts[1].strip()
    else:
        # If the output could not be split, there was no final answer
        final_answer = "No final answer provided."

    return final_answer